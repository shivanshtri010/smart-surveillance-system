import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import sqlite3

# Set page configuration
st.set_page_config(page_title="License Plate Detection Results", layout="wide")

def load_data():
    try:
        conn = sqlite3.connect('vehicle_data.db')
        query = "SELECT * FROM vehicle_data"
        df = pd.read_sql_query(query, conn)
        conn.close()

        df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
        df['outtime'] = pd.to_datetime(df['outtime'], errors='coerce')
        df['In Hour'] = df['intime'].dt.hour
        df['Out Hour'] = df['outtime'].dt.hour
        df['Date'] = df['intime'].dt.date
        df = df.dropna(subset=['Date'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.sort_values('intime').reset_index(drop=True)
        df['Serial Number'] = range(1, len(df) + 1)
        return df
    except sqlite3.Error as e:
        st.error(f"An error occurred while connecting to the database: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def create_plot(df, selected_date):
    df_day = df[df['Date'] == selected_date]
    in_counts = df_day['In Hour'].value_counts().sort_index()
    out_counts = df_day['Out Hour'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(in_counts.index - 0.2, in_counts.values, width=0.4, label='Entries', color='blue')
    ax.bar(out_counts.index + 0.2, out_counts.values, width=0.4, label='Exits', color='red')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{i:02d}:00' for i in range(0, 24, 2)], rotation=45, ha='right')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Number of Vehicles')
    ax.set_title(f'Vehicle Entries and Exits on {selected_date}')
    ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    return fig

def display_dashboard():
    st.title("License Plate Detection Results")

    df = load_data()
    if df is not None and not df.empty:
        selected_date = df['Date'].max()
        df_day = df[df['Date'] == selected_date]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Entry/Exit Graph")
            if not df_day.empty:
                fig = create_plot(df, selected_date)
                st.pyplot(fig)
            else:
                st.write("No data available for the selected date.")

        with col2:
            st.subheader("Security Guard Dashboard")
            display_df = df_day[['Serial Number', 'license_plate', 'vehicle_type', 'intime', 'outtime']]
            display_df.columns = ['Serial Number', 'Vehicle License Plate', 'Vehicle Type', 'Vehicle Intime',
                                  'Vehicle Outtime']
            st.dataframe(display_df,
                         hide_index=True,
                         use_container_width=True,
                         height=300)

        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.write("No data available. Please check your database connection.")

def main():
    placeholder = st.empty()

    while True:
        with placeholder.container():
            display_dashboard()
        time.sleep(1)  # Wait for 5 seconds before updating again

if __name__ == "__main__":
    main()
