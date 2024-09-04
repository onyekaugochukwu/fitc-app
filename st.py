import streamlit as st
import numpy as np
import pandas as pd
import io
import base64
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openai
from openai import OpenAI
import PyPDF2
import os


def main():
    st.set_page_config(layout="wide", page_title="CAAT PRO")
    st.title("Computer Assisted Audit Tool")

    # Initialize session state
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = {}

    # Sidebar for main navigation
    st.sidebar.title("Main Functions")
    main_page = st.sidebar.selectbox("Select Function", 
        ["Data Upload", "Data Management", "Analysis", "Reporting", "Utilities", "AI Analysis"])

    # Main content area
    if main_page == "Data Upload":
        data_upload()
    elif main_page == "Data Management":
        data_management_page()
    elif main_page == "Analysis":
        analysis_page()
    elif main_page == "Reporting":
        reporting_page()
    elif main_page == "Utilities":
        utilities_page()
    elif main_page == "AI Analysis":
        financial_statement_analysis()


def data_upload():
    st.subheader("Data Upload")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Excel file
                data = pd.read_excel(uploaded_file)
            
            st.success(f"Uploaded {uploaded_file.name} successfully!")
            
            # Display data info
            st.write("Data Information:")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
            
            # Display data preview
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # User-specified data types
            st.subheader("Confirm Column Data Types")
            col_types = {}
            for column in data.columns:
                col_type = st.selectbox(
                    f"Select data type for '{column}'",
                    options=['object', 'int64', 'float64', 'datetime64', 'bool'],
                    index=0,
                    key=f"dtype_{column}"
                )
                col_types[column] = col_type
            
            # Apply user-specified data types
            for col, dtype in col_types.items():
                if dtype == 'datetime64':
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                else:
                    data[col] = data[col].astype(dtype)
            
            # Store the data in session state
            st.session_state['uploaded_data'] = {uploaded_file.name: data}
            st.session_state['column_dtypes'] = col_types
            
            st.success("Data types updated successfully!")
            
        except Exception as e:
            st.error(f"An error occurred while reading the file: {str(e)}")

    return uploaded_file is not None


def financial_statement_analysis():
    st.subheader("AI-Driven Financial Statement Analysis")

    # API Key Input
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        openai.api_key = api_key
    elif 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        st.warning("Please enter your OpenAI API Key to proceed.")
        return

    uploaded_file = st.file_uploader("Upload financial statements (PDF)", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            # Create a prompt for the AI to parse and analyze the data
            prompt = f"""
            The following is the text content of a financial report containing various financial statements. Please perform the following tasks:

            1. Identify and extract key financial metrics from all available statements (e.g., Income Statement, Balance Sheet, Cash Flow Statement).
            2. Calculate and provide the following financial ratios:
               - Return on Assets (ROA)
               - Return on Equity (ROE)
               - Debt to Equity Ratio
               - Profit Margin
               - Current Ratio
               - Quick Ratio
            3. Calculate the Beneish M-score to assess the likelihood of financial statement manipulation.
            4. Perform a trend analysis on key metrics over the available time periods.
            5. Provide insights on the overall financial health of the company.
            6. Identify any red flags or areas of concern.
            7. Offer recommendations for improvement.

            Present your analysis in a clear, structured format suitable for a financial report.

            Financial Report Text:
            {pdf_text}
            """
            
            client = OpenAI(api_key=api_key)

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,
                n=1,
                temperature=0.5,
            )

            # Display AI analysis
            st.write("AI-Assisted Financial Statement Analysis:")
            analysis = response.choices[0].message.content
            st.markdown(analysis)

            # Option to download the analysis
            st.download_button(
                label="Download analysis as TXT",
                data=analysis,
                file_name="financial_statement_analysis.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"An error occurred during the analysis: {str(e)}")

    else:
        st.info("Please upload a PDF containing the financial statements to begin the analysis.")


def data_management_page():
    st.header("Data Management")
    if st.session_state['uploaded_data']:
        options = st.multiselect("Select operations", ["Data Profile", "Summarization", "Aging Analysis", "Merge Sheets", "Extract Duplicates", "Compare Sheets"])
        
        if "Data Profile" in options:
            data_profile()
        if "Summarization" in options:
            summarization()
        if "Aging Analysis" in options:
            age_analysis()
        if "Merge Sheets" in options:
            merge_sheets()
        if "Extract Duplicates" in options:
            extract_duplicates()
        if "Compare Sheets" in options:
            compare_sheets()
    else:
        st.warning("Please upload data in the Data Upload section before performing data management operations.")

def analysis_page():
    st.header("Analysis")
    if st.session_state['uploaded_data']:
        options = st.multiselect("Select operations", ["Benford Analysis", "Top-Bottom Analysis", "Find Gaps", "Stratify Data", "Sample Data"])
        
        if "Benford Analysis" in options:
            benford_analysis()
        if "Top-Bottom Analysis" in options:
            summarization()
        if "Find Gaps" in options:
            gap_analysis()
        if "Stratify Data" in options:
            stratify_data()
        if "Sample Data" in options:
            sample_data()
    else:
        st.warning("Please upload data in the Data Upload section before performing analysis.")

def reporting_page():
    st.header("Reporting")
    if st.session_state['uploaded_data']:
        st.write("Reporting functions will be implemented here.")
    else:
        st.warning("Please upload data in the Data Upload section before generating reports.")

def utilities_page():
    st.header("Utilities")
    st.write("Utilities functions will be implemented here.")

def data_profile():
    st.subheader("Data Profile")
    for filename, data in st.session_state['uploaded_data'].items():
        st.write(f"Summary for {filename}:")
        st.write(data.describe())


def summarization():
    st.subheader("Summarization")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing summarization.")
        return

    selected_dataset = st.selectbox("Select dataset to summarize", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset]

    st.subheader("Group Summary Options")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Columns To Analyze")
        group_by = st.multiselect("Group by", df.columns)
        
        # Date grouping options
        date_columns = [col for col, dtype in st.session_state['column_dtypes'].items() if dtype == 'datetime64']
        if date_columns:
            use_date_grouping = st.checkbox("Use date grouping")
            if use_date_grouping:
                date_group_options = st.multiselect("Group dates by", ["Year", "Quarter", "Month", "Week", "Day"])
                date_column = st.selectbox("Select date column for grouping", date_columns)

        sort_order = st.radio("Sort order", ["Ascending", "Descending"])

    with col2:
        st.write("Columns to Summarize")
        summarize_columns = st.multiselect("Select columns", df.columns)

    st.subheader("Statistics Options")
    stats_options = st.multiselect("Select statistics", 
                                   ["Count", "Sum", "Mean", "Median", "Mode", "Min", "Max", "Std Dev"])

    if st.button("Generate Summary"):
        if not group_by and not (date_columns and use_date_grouping and date_group_options):
            st.warning("Please select grouping columns or date grouping options.")
        elif not summarize_columns or not stats_options:
            st.warning("Please select summary columns and statistics options.")
        else:
            # Apply date grouping if selected
            if date_columns and use_date_grouping and date_group_options:
                for option in date_group_options:
                    if option == "Year":
                        df[f"{date_column}_Year"] = df[date_column].dt.year
                    elif option == "Quarter":
                        df[f"{date_column}_Quarter"] = df[date_column].dt.to_period("Q")
                    elif option == "Month":
                        df[f"{date_column}_Month"] = df[date_column].dt.to_period("M")
                    elif option == "Week":
                        df[f"{date_column}_Week"] = df[date_column].dt.to_period("W")
                    elif option == "Day":
                        df[f"{date_column}_Day"] = df[date_column].dt.date
                group_by.extend([f"{date_column}_{option}" for option in date_group_options])

            # Group by the specified columns and aggregate
            summary = df.groupby(group_by)

            agg_dict = {}
            for col in summarize_columns:
                agg_dict[col] = []
                for stat in stats_options:
                    if stat == "Count":
                        agg_dict[col].append(lambda x: x.count())
                    elif stat == "Sum":
                        agg_dict[col].append(lambda x: x.sum())
                    elif stat == "Mean":
                        agg_dict[col].append(lambda x: x.mean())
                    elif stat == "Median":
                        agg_dict[col].append(lambda x: x.median())
                    elif stat == "Mode":
                        agg_dict[col].append(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                    elif stat == "Min":
                        agg_dict[col].append(lambda x: x.min())
                    elif stat == "Max":
                        agg_dict[col].append(lambda x: x.max())
                    elif stat == "Std Dev":
                        agg_dict[col].append(lambda x: x.std())

            result = summary.agg(agg_dict)

            result.columns = ['_'.join(col).strip() for col in result.columns.values]
            result = result.sort_index(ascending=(sort_order == "Ascending"))

            st.session_state['summary_result'] = result
            st.session_state['summary_params'] = {
                'selected_dataset': selected_dataset,
                'group_by': group_by,
                'summarize_columns': summarize_columns,
                'stats_options': stats_options,
                'sort_order': sort_order,
                'use_date_grouping': use_date_grouping if date_columns else False,
                'date_group_options': date_group_options if date_columns and use_date_grouping else None,
                'date_column': date_column if date_columns and use_date_grouping else None
            }
            st.success("Summary generated successfully!")

    if 'summary_result' in st.session_state:
        st.write("Summary Result:")
        st.dataframe(st.session_state['summary_result'])

        st.subheader("Export Options")
        export_format = st.radio("Select export format", ["CSV", "Excel"])
        export_type = st.radio("Select export type", ["Summary Only", "Summary with Detailed Groups"])
        
        if st.button("Prepare Export"):
            if export_type == "Summary Only":
                if export_format == "CSV":
                    csv = st.session_state['summary_result'].to_csv(index=True)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="summary_result.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        st.session_state['summary_result'].to_excel(writer, sheet_name='Summary', index=True)
                    b64 = base64.b64encode(output.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="summary_result.xlsx">Download Excel File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                zip_buffer = io.BytesIO()
                with ZipFile(zip_buffer, 'w') as zip_file:
                    # Add summary to ZIP
                    if export_format == "CSV":
                        zip_file.writestr("summary.csv", st.session_state['summary_result'].to_csv(index=True))
                    else:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            st.session_state['summary_result'].to_excel(writer, sheet_name='Summary', index=True)
                        zip_file.writestr("summary.xlsx", excel_buffer.getvalue())
                    
                    # Add detailed group data to ZIP
                    df = st.session_state['uploaded_data'][st.session_state['summary_params']['selected_dataset']]
                    group_by = st.session_state['summary_params']['group_by']
                    for name, group in df.groupby(group_by):
                        if isinstance(name, tuple):
                            file_name = "_".join(str(n) for n in name)
                        else:
                            file_name = str(name)
                        if export_format == "CSV":
                            zip_file.writestr(f"group_{file_name}.csv", group.to_csv(index=False))
                        else:
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                group.to_excel(writer, sheet_name='Group Data', index=False)
                            zip_file.writestr(f"group_{file_name}.xlsx", excel_buffer.getvalue())

                zip_buffer.seek(0)
                b64 = base64.b64encode(zip_buffer.getvalue()).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="summary_and_groups.zip">Download ZIP File</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.success("Export prepared successfully. Click the link above to download.")
            st.info("Note: Downloading will refresh the page. Use the 'Regenerate Previous Summary' button below to quickly recreate your summary.")

    # Button to regenerate previous summary
    if 'summary_params' in st.session_state:
        if st.button("Regenerate Previous Summary"):
            params = st.session_state['summary_params']
            df = st.session_state['uploaded_data'][params['selected_dataset']]
            
            # Reapply date grouping if applicable
            if params['use_date_grouping'] and params['date_group_options']:
                for option in params['date_group_options']:
                    if option == "Year":
                        df[f"{params['date_column']}_Year"] = df[params['date_column']].dt.year
                    elif option == "Quarter":
                        df[f"{params['date_column']}_Quarter"] = df[params['date_column']].dt.to_period("Q")
                    elif option == "Month":
                        df[f"{params['date_column']}_Month"] = df[params['date_column']].dt.to_period("M")
                    elif option == "Week":
                        df[f"{params['date_column']}_Week"] = df[params['date_column']].dt.to_period("W")
                    elif option == "Day":
                        df[f"{params['date_column']}_Day"] = df[params['date_column']].dt.date

            # Group by the specified columns and aggregate
            summary = df.groupby(params['group_by'])

            agg_dict = {}
            for col in params['summarize_columns']:
                agg_dict[col] = []
                for stat in params['stats_options']:
                    if stat == "Count":
                        agg_dict[col].append(lambda x: x.count())
                    elif stat == "Sum":
                        agg_dict[col].append(lambda x: x.sum())
                    elif stat == "Mean":
                        agg_dict[col].append(lambda x: x.mean())
                    elif stat == "Median":
                        agg_dict[col].append(lambda x: x.median())
                    elif stat == "Mode":
                        agg_dict[col].append(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                    elif stat == "Min":
                        agg_dict[col].append(lambda x: x.min())
                    elif stat == "Max":
                        agg_dict[col].append(lambda x: x.max())
                    elif stat == "Std Dev":
                        agg_dict[col].append(lambda x: x.std())

            result = summary.agg(agg_dict)

            result.columns = ['_'.join(col).strip() for col in result.columns.values]
            result = result.sort_index(ascending=(params['sort_order'] == "Ascending"))

            st.session_state['summary_result'] = result
            st.success("Previous summary regenerated successfully!")
            st.dataframe(result)

def age_analysis():
    st.subheader("Age Analysis")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing age analysis.")
        return

    selected_dataset = st.selectbox("Select dataset for age analysis", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset].copy()

    # Date column selection
    date_columns = [col for col, dtype in st.session_state['column_dtypes'].items() if dtype == 'datetime64']
    if not date_columns:
        st.warning("No date columns found in the dataset. Please ensure you have a date column.")
        return

    date_column = st.selectbox("Select date column for analysis", date_columns)

    # Convert string dates back to datetime for processing
    df[date_column] = pd.to_datetime(df[date_column])

    # Analysis date selection
    max_date = df[date_column].max()
    analysis_date = st.date_input("Analyze As Of", value=max_date, max_value=max_date)
    analysis_date = pd.Timestamp(analysis_date)  # Convert to pandas Timestamp

    # Age intervals
    st.subheader("Age Intervals (Days)")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    intervals = [
        col1.number_input("Interval 1", value=30, min_value=1, step=1),
        col2.number_input("Interval 2", value=60, min_value=1, step=1),
        col3.number_input("Interval 3", value=90, min_value=1, step=1),
        col4.number_input("Interval 4", value=120, min_value=1, step=1),
        col5.number_input("Interval 5", value=150, min_value=1, step=1),
        col6.number_input("Interval 6", value=180, min_value=1, step=1)
    ]

    # Amount column selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        amount_column = st.selectbox("Select amount column to total", numeric_columns)
    else:
        st.warning("No numeric columns found in the dataset.")
        return

    # Grouping column selection
    grouping_column = st.selectbox("Select column to group by (optional)", ["None"] + list(df.columns))

    if st.button("Generate Age Analysis"):
        # Calculate age
        df['Age'] = (analysis_date - df[date_column]).dt.days

        # Define age bins
        bins = [-float('inf')] + [0] + intervals + [float('inf')]
        labels = [f"<= 0"] + [f"<= {i}" for i in intervals] + [f"{intervals[-1]}+"]

        # Categorize ages
        df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

        # Group by age category and optional grouping column
        if grouping_column != "None":
            grouped = df.groupby([grouping_column, 'Age_Category'])
        else:
            grouped = df.groupby('Age_Category')

        # Aggregate results
        results = grouped.agg({
            amount_column: ['count', 'sum'],
            'Age_Category': 'count'
        }).reset_index()

        # Rename columns
        results.columns = ['Group', 'Interval', '# Items', 'Amount', 'Total Items'] if grouping_column != "None" else ['Interval', '# Items', 'Amount', 'Total Items']

        # Calculate percentages
        results['% Total Items'] = results['# Items'] / results['Total Items'].sum() * 100
        results['% Total Amount'] = results['Amount'] / results['Amount'].sum() * 100

        # Display results
        st.write("Age Analysis Results:")
        st.dataframe(results)

        # Save results to session state for potential export and chart generation
        st.session_state['age_analysis_results'] = results

    # Generate chart
    if 'age_analysis_results' in st.session_state:
        show_chart = st.checkbox("Show Age Analysis Chart", value=True)
        if show_chart:
            results = st.session_state['age_analysis_results']
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Interval', y='# Items', data=results, ax=ax)
            ax.set_title("Age Analysis: # of Items By Date")
            ax.set_xlabel("Age")
            ax.set_ylabel("# of Items")
            st.pyplot(fig)

    # Export options
    if 'age_analysis_results' in st.session_state:
        st.subheader("Export Options")
        export_format = st.radio("Select export format", ["CSV", "Excel"])
        
        if st.button("Prepare Export"):
            if export_format == "CSV":
                csv = st.session_state['age_analysis_results'].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="age_analysis_results.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state['age_analysis_results'].to_excel(writer, sheet_name='Age Analysis', index=False)
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="age_analysis_results.xlsx">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.success("Export prepared successfully. Click the link above to download.")


def extract_duplicates():
    st.subheader("Duplicate Item Analysis")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing duplicate analysis.")
        return

    selected_dataset = st.selectbox("Select dataset for analysis", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset].copy()

    # Duplicate analysis options
    analysis_type = st.radio("Select analysis type", ["Row-level duplicates", "Column-level duplicates"])

    if analysis_type == "Column-level duplicates":
        columns_to_analyze = st.multiselect("Select columns to analyze for duplicates", df.columns)
    else:
        columns_to_analyze = df.columns.tolist()

    # Duplicate handling options
    duplicate_option = st.radio("Select duplicate handling option", 
                                ["Tag Duplicates", "Extract Duplicates", 
                                 "Remove Duplicates - Keep First", "Remove Duplicates - Keep Last",
                                 "Extract Non Duplicates"])

    if st.button("Perform Duplicate Analysis"):
        if analysis_type == "Row-level duplicates":
            result = handle_row_duplicates(df, columns_to_analyze, duplicate_option)
        else:
            result = handle_column_duplicates(df, columns_to_analyze, duplicate_option)

        if result is not None:
            st.write("Analysis Result:")
            st.dataframe(result)
            
            # Option to download the result
            csv = result.to_csv(index=False)
            st.download_button(
                label="Download result as CSV",
                data=csv,
                file_name="duplicate_analysis_result.csv",
                mime="text/csv",
            )
        else:
            st.write("No duplicates found or no action taken.")

def handle_row_duplicates(df, columns, option):
    if option == "Tag Duplicates":
        df['Is_Duplicate'] = df.duplicated(subset=columns, keep=False)
        return df
    elif option == "Extract Duplicates":
        return df[df.duplicated(subset=columns, keep=False)]
    elif option == "Remove Duplicates - Keep First":
        return df.drop_duplicates(subset=columns, keep='first')
    elif option == "Remove Duplicates - Keep Last":
        return df.drop_duplicates(subset=columns, keep='last')
    elif option == "Extract Non Duplicates":
        return df[~df.duplicated(subset=columns, keep=False)]
    return None

def handle_column_duplicates(df, columns, option):
    results = []
    for column in columns:
        if option == "Tag Duplicates":
            df[f'{column}_Is_Duplicate'] = df[column].duplicated(keep=False)
        elif option == "Extract Duplicates":
            duplicates = df[df[column].duplicated(keep=False)]
            if not duplicates.empty:
                results.append(duplicates[[column]])
        elif option == "Remove Duplicates - Keep First":
            df = df.drop_duplicates(subset=[column], keep='first')
        elif option == "Remove Duplicates - Keep Last":
            df = df.drop_duplicates(subset=[column], keep='last')
        elif option == "Extract Non Duplicates":
            non_duplicates = df[~df[column].duplicated(keep=False)]
            if not non_duplicates.empty:
                results.append(non_duplicates[[column]])
    
    if option == "Tag Duplicates":
        return df
    elif option in ["Extract Duplicates", "Extract Non Duplicates"]:
        return pd.concat(results, axis=1) if results else None
    elif option.startswith("Remove Duplicates"):
        return df
    return None

            
def benford_analysis():
    st.subheader("Benford's Law Analysis")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing Benford's Law analysis.")
        return

    selected_dataset = st.selectbox("Select dataset for analysis", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset].copy()

    # Column selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns found in the dataset.")
        return

    column_to_analyze = st.selectbox("Column To Analyze", numeric_columns)

    # Digital Tests
    st.subheader("Digital Tests")
    col1, col2 = st.columns(2)
    with col1:
        first_digit_test = st.checkbox("First Digit Test", value=True)
        second_digit_test = st.checkbox("Second Digit Test")
        first_two_digits_test = st.checkbox("First 2 Digits Test")
    with col2:
        first_three_digits_test = st.checkbox("First 3 Digits Test")
        last_two_digits_test = st.checkbox("Last 2 Digits Test")
        number_duplication_test = st.checkbox("Number Duplication Test")

    # Group options
    st.subheader("Groups")
    group_by = st.selectbox("Group by", ["None"] + list(df.columns))
    sort_order = st.radio("Sort order", ["Ascending", "Descending"])

    # Additional options
    include_stratified_analysis = st.checkbox("Include Stratified Analysis")
    if include_stratified_analysis:
        num_strata = st.number_input("Number of strata", min_value=2, max_value=10, value=5)
    else:
        num_strata = None
    chart_results = st.checkbox("Chart Results", value=True)
    condense_output = st.checkbox("Condense Output")

    if st.button("Perform Benford Analysis"):
        try:
            results = perform_benford_analysis(df[column_to_analyze], 
                                               first_digit_test, 
                                               second_digit_test, 
                                               first_two_digits_test, 
                                               first_three_digits_test, 
                                               last_two_digits_test, 
                                               number_duplication_test,
                                               group_by,
                                               df[group_by] if group_by != "None" else None,
                                               sort_order,
                                               include_stratified_analysis,
                                               num_strata)
            
            if results:
                st.session_state['benford_results'] = results
                display_results(results, chart_results, condense_output)
            else:
                st.warning("No valid results were produced. The dataset might be empty or contain invalid values.")
        except Exception as e:
            st.error(f"An error occurred during the analysis: {str(e)}")

def perform_benford_analysis(data, first_digit, second_digit, first_two, first_three, last_two, duplication, group_by, group_data, sort_order, stratified, num_strata):
    if data.empty:
        return None

    results = {}

    if group_by is not None and group_by != "None":
        grouped = data.groupby(group_data)
        for name, group in grouped:
            if not group.empty:
                results[name] = analyze_group(group, first_digit, second_digit, first_two, first_three, last_two, duplication)
    else:
        results['Overall'] = analyze_group(data, first_digit, second_digit, first_two, first_three, last_two, duplication)

    if stratified and num_strata:
        results['Stratified'] = perform_stratified_analysis(data, num_strata, first_digit, second_digit, first_two, first_three, last_two, duplication)

    return results if results else None

def analyze_group(data, first_digit, second_digit, first_two, first_three, last_two, duplication):
    group_results = {}

    if first_digit:
        group_results['First Digit'] = analyze_digits(data, 1)
    if second_digit:
        group_results['Second Digit'] = analyze_digits(data, 2)
    if first_two:
        group_results['First Two Digits'] = analyze_digits(data, 1, 2)
    if first_three:
        group_results['First Three Digits'] = analyze_digits(data, 1, 3)
    if last_two:
        group_results['Last Two Digits'] = analyze_digits(data, -2)
    if duplication:
        group_results['Number Duplication'] = analyze_duplication(data)

    return group_results

def analyze_digits(data, start, end=None):
    if end is None:
        end = start + 1
    
    # Convert to string and remove decimal points
    str_data = data.abs().astype(str).str.replace('.', '', regex=False)
    
    # Pad with zeros to ensure we have at least 'end' digits
    str_data = str_data.str.pad(end, side='left', fillchar='0')
    
    # Extract the required digits
    digits = str_data.str[start-1:end].astype(int)
    
    observed_freq = digits.value_counts(normalize=True).sort_index()
    
    if start == 1 and end == 2:  # First digit
        expected_freq = pd.Series([np.log10(1 + 1/d) for d in range(1, 10)], index=range(1, 10))
    elif end - start == 1:  # Single digit
        expected_freq = pd.Series([0.1 for _ in range(10)], index=range(10))
    else:  # Multiple digits
        expected_freq = pd.Series([1/100 for _ in range(100)], index=range(100))

    # Align observed and expected frequencies
    observed_freq, expected_freq = observed_freq.align(expected_freq, fill_value=0)
    
    # Remove any zero frequencies to avoid division by zero in chi-square test
    mask = (observed_freq != 0) & (expected_freq != 0)
    observed_freq = observed_freq[mask]
    expected_freq = expected_freq[mask]

    # Normalize expected frequencies to match the sum of observed frequencies
    if not observed_freq.empty and not expected_freq.empty:
        expected_freq = expected_freq * (observed_freq.sum() / expected_freq.sum())
        chi_square, p_value = stats.chisquare(observed_freq, expected_freq)
    else:
        chi_square, p_value = np.nan, np.nan

    return {
        'observed': observed_freq,
        'expected': expected_freq,
        'chi_square': chi_square,
        'p_value': p_value
    }

def analyze_duplication(data):
    duplicates = data.duplicated(keep=False)
    return duplicates.value_counts(normalize=True)

def perform_stratified_analysis(data, num_strata, first_digit, second_digit, first_two, first_three, last_two, duplication):
    strata = pd.qcut(data, q=num_strata, labels=False)
    stratified_results = {}

    for i in range(num_strata):
        stratum_data = data[strata == i]
        stratum_name = f"Stratum {i+1}"
        stratified_results[stratum_name] = analyze_group(stratum_data, first_digit, second_digit, first_two, first_three, last_two, duplication)

    return stratified_results

def display_results(results, chart_results, condense_output):
    for group, tests in results.items():
        st.write(f"### Results for {group}")
        for test, data in tests.items():
            st.write(f"#### {test}")
            if isinstance(data, dict) and 'observed' in data:
                df_result = pd.DataFrame({
                    'Observed': data['observed'],
                    'Expected': data['expected']
                })
                
                if df_result.empty:
                    st.write("No valid data for analysis.")
                else:
                    st.write(df_result)
                    st.write(f"Chi-square statistic: {data['chi_square']:.4f}")
                    st.write(f"p-value: {data['p_value']:.4f}")

                    if chart_results and not df_result.empty:
                        fig, ax = plt.subplots()
                        df_result.plot(kind='bar', ax=ax)
                        ax.set_title(f"{test} - {group}")
                        ax.set_xlabel("Digit")
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
            else:
                st.write(data)

    if not condense_output:
        st.write("Detailed output (not condensed)")
        st.json(results)

def gap_analysis():
    st.subheader("Gap Analysis")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing gap analysis.")
        return

    selected_dataset = st.selectbox("Select dataset for analysis", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset].copy()

    # Column selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns found in the dataset.")
        return

    column_to_analyze = st.selectbox("Select Column To Analyze", ["None"] + numeric_columns)

    if column_to_analyze != "None":
        # Gap analysis parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            start_value = st.number_input("Start Value", value=df[column_to_analyze].min())
        with col2:
            end_value = st.number_input("End Value", value=df[column_to_analyze].max())
        with col3:
            increment = st.number_input("Increment", value=1, min_value=1)

        # Options
        show_missing_only = st.checkbox("Show missing items only", value=True)
        ignore_duplicates = st.checkbox("Ignore duplicate items")
        remove_duplicates = st.checkbox("Remove duplicate items from output")

        if st.button("Perform Gap Analysis"):
            # Perform gap analysis
            gaps = find_gaps(df[column_to_analyze], start_value, end_value, increment, 
                             show_missing_only, ignore_duplicates, remove_duplicates)
            
            if gaps.empty:
                st.write("No gaps found in the selected range.")
            else:
                st.write("Gap Analysis Results:")
                st.dataframe(gaps)

                # Option to download the result
                csv = gaps.to_csv(index=False)
                st.download_button(
                    label="Download result as CSV",
                    data=csv,
                    file_name="gap_analysis_result.csv",
                    mime="text/csv",
                )

        # Preview
        st.subheader("Preview")
        preview_df = pd.DataFrame({column_to_analyze: range(int(start_value), int(end_value) + 1, int(increment))})
        st.dataframe(preview_df)

def find_gaps(series, start, end, increment, show_missing_only, ignore_duplicates, remove_duplicates):
    # Create a range of expected values
    expected_range = pd.Series(np.arange(start, end + increment, increment))
    
    if ignore_duplicates or remove_duplicates:
        series = series.drop_duplicates()
    
    # Find missing values
    missing = expected_range[~expected_range.isin(series)]
    
    # Create result dataframe
    if show_missing_only:
        result = pd.DataFrame({'Missing Values': missing})
    else:
        result = pd.DataFrame({'Values': expected_range})
        result['Status'] = np.where(result['Values'].isin(series), 'Present', 'Missing')
    
    return result

def stratify_data():
    st.subheader("Stratified Analysis")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing stratified analysis.")
        return

    selected_dataset = st.selectbox("Select dataset for analysis", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset].copy()

    # Column selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("At least two numeric columns are required for stratified analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        column_to_stratify = st.selectbox("Column To Stratify", numeric_columns)
    with col2:
        column_to_total = st.selectbox("Column To Total", numeric_columns)

    # Stratification options
    st.subheader("Stratification Options")
    strata_method = st.radio("Strata Creation Method", 
                             ["Fixed band size", "Equal-sized bands"])
    
    if strata_method == "Fixed band size":
        num_strata = st.number_input("Number of strata", min_value=2, value=5)
        band_size = (df[column_to_stratify].max() - df[column_to_stratify].min()) / num_strata
    else:
        num_strata = st.number_input("Number of equal-sized bands", min_value=2, value=5)

    # Additional options
    chart_results = st.checkbox("Chart Results")
    show_sample_column = st.checkbox("Show Sample Column")
    
    if st.button("Perform Stratified Analysis"):
        if strata_method == "Fixed band size":
            df['Strata'] = pd.cut(df[column_to_stratify], bins=num_strata, labels=range(1, num_strata+1))
        else:
            df['Strata'] = pd.qcut(df[column_to_stratify], q=num_strata, labels=range(1, num_strata+1))
        
        # Calculate statistics for each stratum
        strata_stats = df.groupby('Strata', observed=True).agg({
            column_to_stratify: ['mean', 'var', 'std'],
            column_to_total: ['count', 'sum', 'mean', 'var', 'std']
        }).reset_index()

        # Flatten the multi-level column index
        strata_stats.columns = [f"{'' if col[0] == 'Strata' else col[0] + '_'}{col[1]}" for col in strata_stats.columns]

        # Ensure the column names match
        column_mapping = {
            'Strata': 'Strata #',
            f'{column_to_stratify}_mean': 'Avg Value',
            f'{column_to_stratify}_var': 'Sample Var',
            f'{column_to_stratify}_std': 'Sample Std Dev',
            f'{column_to_total}_count': 'Count',
            f'{column_to_total}_sum': 'Total',
            f'{column_to_total}_mean': 'Avg Total',
            f'{column_to_total}_var': 'Pop Var',
            f'{column_to_total}_std': 'Pop Std Dev'
        }

        # Only rename the columns that exist in the DataFrame
        strata_stats.rename(columns={col: column_mapping[col] for col in strata_stats.columns if col in column_mapping}, inplace=True)

        # Calculate population statistics
        pop_stats_stratify = df[column_to_stratify].agg(['count', 'min', 'max', 'mean', 'var', 'std'])
        pop_stats_total = df[column_to_total].agg(['sum', 'mean', 'var', 'std'])

        # Concatenate the statistics
        pop_stats = pd.concat([pop_stats_stratify, pop_stats_total])
        pop_stats = pop_stats.reset_index()

        # Debug: Print the structure of pop_stats
        st.write("Shape of pop_stats:", pop_stats.shape)
        st.write("Columns in pop_stats before renaming:", pop_stats.columns)

        # Ensure renaming matches the number of columns
        if pop_stats.shape[1] == 2:  # Only proceed if there are two columns
            pop_stats.columns = ['Statistic', 'Value']
        else:
            st.warning("Unexpected number of columns in pop_stats. Skipping renaming.")

        # Display results
        st.write("Stratified Analysis Results:")
        st.dataframe(strata_stats)
        
        st.write("\nPopulation Statistics:")
        st.dataframe(pop_stats)

        if chart_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of stratified column
            ax1.hist(df[column_to_stratify], bins=num_strata)
            ax1.set_title(f"Distribution of {column_to_stratify}")
            ax1.set_xlabel(column_to_stratify)
            ax1.set_ylabel("Frequency")

            # Box plot of total column by strata
            df.boxplot(column=column_to_total, by='Strata', ax=ax2)
            ax2.set_title(f"{column_to_total} by Strata")
            ax2.set_xlabel("Strata")
            ax2.set_ylabel(column_to_total)

            st.pyplot(fig)

        if show_sample_column:
            sample_size = st.number_input("Sample size per stratum", min_value=1, value=10)
            sample_df = df.groupby('Strata').apply(lambda x: x.sample(min(len(x), sample_size))).reset_index(drop=True)
            st.write("\nSample Data:")
            st.dataframe(sample_df)

        # Option to download the results
        csv_strata = strata_stats.to_csv(index=False)
        st.download_button(
            label="Download strata results as CSV",
            data=csv_strata,
            file_name="stratified_analysis_results.csv",
            mime="text/csv",
        )

        csv_pop = pop_stats.to_csv(index=False)
        st.download_button(
            label="Download population statistics as CSV",
            data=csv_pop,
            file_name="population_statistics.csv",
            mime="text/csv",
        )


def sample_data():
    st.subheader("Sampling")

    if not st.session_state.get('uploaded_data'):
        st.warning("Please upload data before performing sampling.")
        return

    selected_dataset = st.selectbox("Select dataset for sampling", list(st.session_state['uploaded_data'].keys()))
    df = st.session_state['uploaded_data'][selected_dataset].copy()

    st.write("Random Sample")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.number_input("# of Sample Items:", min_value=1, max_value=len(df), value=6)
        sample_start = st.number_input("Sample Between Rows:", min_value=1, max_value=len(df), value=1)
    with col2:
        sample_end = st.number_input("And:", min_value=sample_start, max_value=len(df), value=min(sample_start + 5000, len(df)))
        random_seed = st.number_input("Random # Seed:", value=73636)

    if st.button("Calculate Sample"):
        # Set random seed
        np.random.seed(random_seed)
        
        # Create sample range
        sample_range = range(sample_start - 1, sample_end)
        
        # Perform sampling
        sample_indices = np.random.choice(sample_range, size=num_samples, replace=False)
        sample_df = df.iloc[sample_indices].copy()
        
        # Add Sample Sequence column
        sample_df['SampleSequence'] = range(1, num_samples + 1)
        
        # Add OriginalRow column
        sample_df['OriginalRow'] = sample_indices + 1  # Adding 1 to match Excel's 1-based indexing
        
        # Reorder columns to match the example output
        columns_order = ['SampleSequence', 'OriginalRow'] + [col for col in sample_df.columns if col not in ['SampleSequence', 'OriginalRow']]
        sample_df = sample_df[columns_order]
        
        # Display results
        st.write("Sample Results:")
        st.dataframe(sample_df)
        
        # Option to download the results
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download sample as CSV",
            data=csv,
            file_name="random_sample_results.csv",
            mime="text/csv",
        )

        # Store the sample in session state for potential further analysis
        st.session_state['sample_result'] = sample_df








def merge_sheets():
    st.subheader("Merge Sheets")
    st.write("Merge functionality will be implemented here.")

def split_sheet():
    st.subheader("Split Sheet")
    st.write("Split functionality will be implemented here.")

def compare_sheets():
    st.subheader("Compare Sheets")
    st.write("Compare functionality will be implemented here.")

if __name__ == "__main__":
    main()