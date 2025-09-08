import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import streamlit as st
import matplotlib.pyplot as plt
from html2image import Html2Image
import re
import pytesseract
from difflib import SequenceMatcher

# ---------------------- CONFIG ----------------------
CATALOG_DIR = Path("catalog")
CATALOG_CSV = CATALOG_DIR / "catalog.csv"
BILL_LOG_PATH = Path("generated_bills/billing_log.csv")
ANALYTICS_PASSWORD = "admin123"  # Change this to your preferred password

st.set_page_config(page_title="Product Catalog Billing System", page_icon="🛒", layout="wide")
tab1, tab2 = st.tabs(["🧾 Billing", "📊 Analytics"])

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_error" not in st.session_state:
    st.session_state.auth_error = False


# ---------------------- Catalog Loading ----------------------
@st.cache_data(show_spinner=False)
def load_catalog_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["item_name", "price", "image_path"])

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["item_name", "price", "image_path"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Required column '{col}' not found in catalog.csv")
            return pd.DataFrame(columns=required_cols)

    # Handle optional columns with defaults
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    if "in_stock" not in df.columns:
        df["in_stock"] = True
    if "cost_price" not in df.columns:
        df["cost_price"] = df["price"] * 0.6  # Default to 60% of price

    # Clean data
    df = df.dropna(subset=["item_name", "price", "image_path"]).copy()

    # Trim whitespace
    for c in ["item_name", "image_path", "category"]:
        df[c] = df[c].astype(str).str.strip()

    # Standardize price/cost
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce").fillna(df["price"] * 0.6)

    # Handle in_stock column
    df["in_stock"] = df["in_stock"].astype(str).str.lower().replace(
        {"true": True, "false": False, "yes": True, "no": False})
    df["in_stock"] = df["in_stock"].fillna(True).astype(bool)

    return df


# ---------------------- Billing App ----------------------
with tab1:
    st.title("🛒 Product Catalog Billing System")

    catalog_df = load_catalog_df(CATALOG_CSV)
    if catalog_df.empty:
        st.warning("Catalog is empty. Add rows to catalog/catalog.csv")
    else:
        # Create lookup dictionaries
        price_lookup = dict(zip(catalog_df["item_name"], catalog_df["price"]))
        cost_lookup = dict(zip(catalog_df["item_name"], catalog_df["cost_price"]))
        stock_lookup = dict(zip(catalog_df["item_name"], catalog_df["in_stock"]))
        image_lookup = dict(zip(catalog_df["item_name"], catalog_df["image_path"]))
        category_lookup = dict(zip(catalog_df["item_name"], catalog_df["category"]))

        # Get unique categories
        categories = ["All"] + sorted(catalog_df["category"].unique().tolist())

    if "cart" not in st.session_state:
        st.session_state.cart: Dict[str, Dict[str, float | int]] = {}

    # Search and filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search products", placeholder="Enter product name...")
    with col2:
        items_per_row = st.selectbox("Show per row", options=[3, 4, 6], index=1)
    with col3:
        selected_category = st.selectbox("Category", options=categories)

    # Filter products based on search and category
    filtered_df = catalog_df.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["item_name"].str.contains(search_query, case=False)]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]

    # Display products in a grid
    st.subheader("📦 Available Products")

    # Create a grid layout for items
    items = filtered_df["item_name"].tolist()
    if not items:
        st.info("No products match your search criteria.")
    else:
        for i in range(0, len(items), items_per_row):
            cols = st.columns(items_per_row)
            for j, col in enumerate(cols):
                if i + j < len(items):
                    item_name = items[i + j]
                    img_path = CATALOG_DIR / image_lookup[item_name]
                    price = price_lookup[item_name]
                    in_stock = stock_lookup[item_name]

                    with col:
                        # Create a card-like container
                        with st.container():
                            # Display product image
                            if img_path.exists():
                                try:
                                    img = Image.open(img_path)
                                    img = img.resize((200, 200))
                                    col.image(img, width=150)
                                except Exception as e:
                                    col.image("https://via.placeholder.com/150x150?text=No+Image", width=150)
                            else:
                                col.image("https://via.placeholder.com/150x150?text=No+Image", width=150)

                            # Display product name and price
                            st.markdown(f"**{item_name}**")
                            st.write(f"**₹{price:.2f}**")

                            # Stock status
                            if in_stock:
                                st.success("In stock")
                            else:
                                st.error("Out of stock")

                            # Add to cart controls
                            if in_stock:
                                # Use a shorter label for the quantity input
                                qty = st.number_input(
                                    "Quantity",
                                    min_value=0,
                                    max_value=100,
                                    value=0,
                                    key=f"qty_{item_name}"
                                )

                                if qty > 0 and st.button(f"Add to Cart", key=f"add_{item_name}"):
                                    cost_price = cost_lookup.get(item_name, 0.0)
                                    st.session_state.cart[item_name] = {
                                        "price": float(price),
                                        "qty": int(qty),
                                        "cost_price": float(cost_price)
                                    }
                                    st.success(f"Added {qty} × {item_name}")
                            else:
                                st.button("Out of Stock", disabled=True)
                        st.markdown("---")  # Divider between products

    st.divider()
    st.subheader("🛒 Cart")

    if st.session_state.cart:
        # Create a list of items for the table
        table_data = []
        for name, rec in st.session_state.cart.items():
            table_data.append({
                "Item": name,
                "Unit Price": rec["price"],
                "Qty": rec["qty"],
                "Total": round(rec["price"] * rec["qty"], 2)
            })

        # Display the table
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        # Add + and - buttons for each item
        for name, rec in st.session_state.cart.items():
            col1, col2, col3, col4 = st.columns([1, 2, 3, 2])
            with col1:
                if st.button("➖", key=f"dec_{name}"):
                    if st.session_state.cart[name]["qty"] > 1:
                        st.session_state.cart[name]["qty"] -= 1
                    else:
                        del st.session_state.cart[name]
                    st.rerun()
            with col2:
                st.text(f"Qty: {st.session_state.cart[name]['qty']}")
            with col3:
                if st.button("➕", key=f"inc_{name}"):
                    st.session_state.cart[name]["qty"] += 1
                    st.rerun()
            with col4:
                new_price = st.number_input(
                    f"Price",
                    min_value=0.0,
                    value=float(st.session_state.cart[name]["price"]),
                    key=f"price_{name}"
                )
                st.session_state.cart[name]["price"] = new_price

            st.markdown("---")

        # Calculate totals
        grand_total = sum(rec["price"] * rec["qty"] for rec in st.session_state.cart.values())

        st.metric("Total (Before Discount)", f"₹{grand_total:,.2f}")

        discount_percent = st.slider("Apply Discount (%)", 0, 50, 0)
        discount_amt = (grand_total * discount_percent) / 100.0
        final_total = grand_total - discount_amt
        if discount_percent > 0:
            st.metric("Discount", f"-₹{discount_amt:,.2f}")
        st.metric("Grand Total (After Discount)", f"₹{final_total:,.2f}")

        st.subheader("🧾 Final Details")
        billed_to = st.text_input("Customer Name", value="Customer", key="billed_to")
        mobile = st.text_input("Mobile Number", key="billed_mobile")

        col1, col2 = st.columns(2)
        if col1.button("✅ Save & Generate Invoice"):
            if not billed_to or not mobile:
                st.error("Please fill both customer name and mobile number.")
            else:
                now = datetime.now()
                output_dir = Path("generated_bills")
                output_dir.mkdir(exist_ok=True)
                csv_filename = now.strftime(f"{billed_to}_%Y%m%d_%H%M%S.csv")
                csv_path = output_dir / csv_filename

                # Create a copy with all details for the log
                log_items = []
                for name, rec in st.session_state.cart.items():
                    cost = cost_lookup.get(name, 0.0)
                    log_items.append({
                        "Item": name,
                        "Unit Price": rec["price"],
                        "Cost Price": cost,
                        "Qty": rec["qty"],
                        "Total": round(rec["price"] * rec["qty"], 2),
                        "Profit": round((rec["price"] - cost) * rec["qty"], 2),
                        "Billed To": billed_to,
                        "Mobile": mobile,
                        "timestamp": now,
                        "Grand Total": final_total,
                        "Discount %": discount_percent if discount_percent > 0 else 0
                    })

                log_df = pd.DataFrame(log_items)

                # Save customer invoice without cost and profit
                customer_items = []
                for name, rec in st.session_state.cart.items():
                    customer_items.append({
                        "Item": name,
                        "Unit Price": rec["price"],
                        "Qty": rec["qty"],
                        "Total": round(rec["price"] * rec["qty"], 2)
                    })

                customer_df = pd.DataFrame(customer_items)
                customer_df["Billed To"] = billed_to
                customer_df["Mobile"] = mobile
                customer_df["timestamp"] = now
                customer_df["Grand Total"] = final_total
                if discount_percent > 0:
                    customer_df["Discount %"] = discount_percent
                customer_df.to_csv(csv_path, index=False)

                html_invoice = customer_df.to_html(index=False)
                discount_line = f"Discount: ₹{discount_amt:,.2f}<br/>" if discount_percent > 0 else ""
                html_template = f"""
                <html>
                <head>
                    <title>Invoice - {billed_to}</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{
                            font-family: 'Segoe UI', sans-serif;
                            padding: 30px;
                            background-color: #f9f9f9;
                            max-width: 900px;
                            margin: auto;
                            border: 2px solid #ccc;
                            border-radius: 8px;
                        }}
                        .header {{ text-align: center; margin-bottom: 30px; }}
                        .header h1 {{ margin: 0; font-size: 2.2rem; color: #2c3e50; }}
                        .header p {{ margin: 4px 0; font-size: 1.1rem; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                        th, td {{ border: 1px solid #ccc; padding: 10px; text-align: left; }}
                        th {{ background-color: #2c3e50; color: white; }}
                        .totals {{ margin-top: 20px; font-size: 1.2rem; font-weight: bold; text-align: right; }}
                        .footer {{ margin-top: 40px; text-align: center; font-style: italic; color: #555; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🧾Shree Shyam FC Palwal</h1>
                        <p><b>Customer:</b> {billed_to} | <b>Mobile:</b> {mobile}</p>
                        <p><b>Date:</b> {now.strftime('%d-%m-%Y')}</p>
                    </div>
                    {html_invoice}
                    <p class="totals">{discount_line}Grand Total: ₹{final_total:,.2f}</p>
                    <div class="footer"><p>Thank you for choosing Shyam FC Palwal!</p></div>
                </body>
                </html>
                """

                html_filename = now.strftime(f"{billed_to}_%Y%m%d_%H%M%S.html")
                html_path = output_dir / html_filename
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_template)

                try:
                    output_dir.mkdir(exist_ok=True)
                    hti = Html2Image(output_path=str(output_dir))
                    png_filename = now.strftime(f"{billed_to}_%Y%m%d_%H%M%S.png")
                    png_path = output_dir / png_filename
                    hti.screenshot(html_file=str(html_path), save_as=png_filename)

                    with open(png_path, "rb") as img_f:
                        st.download_button(
                            label="📥 Download Invoice Image (PNG)",
                            data=img_f,
                            file_name=png_filename,
                            mime="image/png"
                        )
                    with open(html_path, "rb") as html_f:
                        st.download_button(
                            label="⬇️ Download Invoice (HTML)",
                            data=html_f,
                            file_name=html_filename,
                            mime="text/html"
                        )
                except Exception as ex:
                    st.warning(f"Could not render PNG from HTML automatically: {ex}")

                # Append to billing_log.csv (with cost and profit data)
                if BILL_LOG_PATH.exists():
                    existing_log = pd.read_csv(BILL_LOG_PATH)
                    df_combined = pd.concat([existing_log, log_df], ignore_index=True)
                else:
                    df_combined = log_df
                df_combined.to_csv(BILL_LOG_PATH, index=False)

                st.success("Invoice saved successfully!")
                st.session_state.cart.clear()
                st.rerun()

        if col2.button("🗑️ Clear Cart"):
            st.session_state.cart.clear()
            st.rerun()
    else:
        st.info("No items in cart.")

# ---------------------- Analytics ----------------------
with tab2:
    st.subheader("📊 Analytics - Admin Access")

    # Password protection - check if already authenticated
    if not st.session_state.authenticated:
        password = st.text_input("Enter password to access analytics", type="password")

        if st.button("Authenticate"):
            if password == ANALYTICS_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.auth_error = False
                st.rerun()
            else:
                st.session_state.auth_error = True
                st.error("Incorrect password. Access denied.")
    else:
        # Show content if authenticated
        if not BILL_LOG_PATH.exists():
            st.warning("No billing history found yet.")
        else:
            @st.cache_data(ttl=600, show_spinner=False)
            def load_billing_log():
                return pd.read_csv(BILL_LOG_PATH, parse_dates=["timestamp"], dayfirst=True)


            df_log = load_billing_log()
            if "timestamp" in df_log.columns:
                df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors="coerce")
                df_log.dropna(subset=["timestamp"], inplace=True)

            # Show all bills
            st.subheader("📋 All Bills")
            if not df_log.empty:
                # Format the display of bills
                bills_summary = df_log.groupby(["Billed To", "Mobile", "timestamp", "Grand Total"]).agg({
                    "Item": "count",
                    "Total": "sum"
                }).reset_index()

                bills_summary.columns = ["Customer", "Mobile", "Date", "Grand Total", "Items Count", "Subtotal"]
                bills_summary = bills_summary.sort_values("Date", ascending=False)

                st.dataframe(
                    bills_summary,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(
                            "Date",
                            format="DD/MM/YYYY HH:mm",
                        ),
                        "Grand Total": st.column_config.NumberColumn(
                            "Grand Total",
                            format="₹%.2f",
                        ),
                        "Subtotal": st.column_config.NumberColumn(
                            "Subtotal",
                            format="₹%.2f",
                        )
                    },
                    use_container_width=True
                )

                # Show detailed bill when a row is selected
                st.subheader("📄 Bill Details")
                if len(bills_summary) > 0:
                    selected_index = st.selectbox(
                        "Select a bill to view details",
                        range(len(bills_summary)),
                        format_func=lambda
                            x: f"{bills_summary.iloc[x]['Customer']} - {bills_summary.iloc[x]['Date'].strftime('%d/%m/%Y %H:%M')} - ₹{bills_summary.iloc[x]['Grand Total']:.2f}"
                    )

                    if selected_index is not None:
                        selected_bill = bills_summary.iloc[selected_index]
                        bill_details = df_log[
                            (df_log["Billed To"] == selected_bill["Customer"]) &
                            (df_log["Mobile"] == selected_bill["Mobile"]) &
                            (df_log["timestamp"] == selected_bill["Date"])
                            ]

                        st.write(f"**Customer:** {selected_bill['Customer']}")
                        st.write(f"**Mobile:** {selected_bill['Mobile']}")
                        st.write(f"**Date:** {selected_bill['Date'].strftime('%d/%m/%Y %H:%M')}")

                        # Display items table
                        items_df = bill_details[["Item", "Unit Price", "Qty", "Total"]]
                        st.dataframe(items_df, use_container_width=True)

                        # Display totals
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Subtotal", f"₹{bill_details['Total'].sum():.2f}")
                        with col2:
                            st.metric("Grand Total", f"₹{selected_bill['Grand Total']:.2f}")

                        if "Discount %" in bill_details.columns and bill_details["Discount %"].iloc[0] > 0:
                            st.write(f"**Discount:** {bill_details['Discount %'].iloc[0]}%")

            total_bills = len(df_log["timestamp"].unique()) if "timestamp" in df_log.columns else 0
            total_sales = df_log["Grand Total"].sum() if "Grand Total" in df_log.columns else 0
            total_profit = df_log["Profit"].sum() if "Profit" in df_log.columns else 0

            # Get top customers
            if "Billed To" in df_log.columns and "Mobile" in df_log.columns:
                top_customers = (
                    df_log.groupby(["Billed To", "Mobile"])
                    .agg({"Grand Total": "sum", "Profit": "sum"})
                    .sort_values(by="Grand Total", ascending=False)
                    .head(5)
                )
            else:
                top_customers = pd.DataFrame()

            st.subheader("📈 Sales Analytics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Bills", total_bills)
            col2.metric("Total Sales", f"₹{total_sales:,.2f}")
            col3.metric("Total Profit", f"₹{total_profit:,.2f}")

            if not top_customers.empty:
                st.markdown("### 🧍‍♂️ Top Customers by Sales")
                st.dataframe(top_customers.rename(columns={"Grand Total": "Total Spend", "Profit": "Total Profit"}))

            if "timestamp" in df_log.columns:
                df_log["Month"] = df_log["timestamp"].dt.to_period("M").astype(str)
                monthly = df_log.groupby("Month").agg({"Grand Total": "sum", "Profit": "sum"}).reset_index()



            st.download_button(
                "⬇️ Download Full Billing Log",
                data=df_log.to_csv(index=False).encode("utf-8"),
                file_name="billing_log.csv",
                mime="text/csv"
            )

        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()