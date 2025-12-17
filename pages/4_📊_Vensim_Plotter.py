import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --- Page config ---
st.set_page_config(page_title="Flexible Vensim Plotter", page_icon="📊", layout="wide")

# --- Load external CSS ---
css_file = Path(__file__).parent.parent / "assets" / "style.css"
if css_file.exists():
    css_content = css_file.read_text()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

st.title("📊 Flexible Vensim Plotter")
st.caption("Create customized time series visualizations from your Vensim temporal data.")

# Data input method selection
input_method = st.radio(
    "Select Input Method",
    options=["Paste Data", "Upload CSV"],
    horizontal=True,
    help="Choose how to input your data"
)

df = None

if input_method == "Paste Data":
    st.markdown("### 📝 Paste Your Data")
    st.markdown("Copy data from Excel/spreadsheet (including headers) and paste it below. Separate columns with tabs or commas.")
    
    # Text area for pasting data
    pasted_data = st.text_area(
        "Paste your data here",
        height=150,
        placeholder="Paste tab-separated or comma-separated data here...\n\nExample:\nVariable\tGroup 1\tGroup 2\t2020\t2021\t2022\nGDP\tEurope\tHigh\t100\t105\t110\nGDP\tAsia\tMedium\t80\t85\t90",
        label_visibility="collapsed"
    )
    
    if pasted_data:
        try:
            # Try to parse as tab-separated first, then comma-separated
            from io import StringIO
            
            # Try tab-separated
            try:
                df = pd.read_csv(StringIO(pasted_data), sep='\t')
                # If only one column, might be comma-separated
                if len(df.columns) == 1:
                    df = pd.read_csv(StringIO(pasted_data), sep=',')
            except:
                # Try comma-separated
                df = pd.read_csv(StringIO(pasted_data), sep=',')
            
            # Show editable preview in dropdown
            with st.expander("📋 Preview & Edit Data", expanded=False):
                edited_df = st.data_editor(
                    df,
                    num_rows="dynamic",
                    use_container_width=True,
                    height=400,
                    key="data_editor"
                )
                df = edited_df.copy()
            
        except Exception as e:
            st.error(f"Could not parse data. Please ensure it's tab or comma-separated. Error: {str(e)}")
            df = None
    else:
        df = None

elif input_method == "Upload CSV":
    st.markdown("### 📂 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Show editable preview in dropdown
        with st.expander("📋 Preview & Edit Data", expanded=False):
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                height=400,
                key="csv_data_editor"
            )
            df = edited_df.copy()

if df is not None:
    try:
        # Data is available, proceed with processing
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows where all values are empty strings
        df = df[~(df.astype(str).apply(lambda x: x.str.strip() == '').all(axis=1))]
        
        st.success(f"✅ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Auto-rename "Time (Year)" to "Variable" if it exists
        if "Time (Year)" in df.columns:
            df = df.rename(columns={"Time (Year)": "Variable"})
        
        # Identify columns
        all_columns = df.columns.tolist()
        
        # Let user identify the structure
        st.subheader("🔧 Configure Data Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            # Smart default: check for "Variable" column
            default_index = 0
            if "Variable" in all_columns:
                default_index = all_columns.index("Variable")
            elif len(all_columns) > 0:
                default_index = 0
            
            variable_col = st.selectbox(
                "Select Variable Column",
                options=all_columns,
                index=default_index,
                help="Column containing variable names"
            )
            
            # Variable parsing option
            parse_variable = st.checkbox(
                "Parse variable names using [] and : separators",
                value=True,
                help="Extract groups from variable names like 'variable1[apple] : china' and auto-fill Group columns"
            )
        
        # Parse variable column if enabled
        if parse_variable:
            import re
            
            def parse_variable_name(name):
                """Parse variable name to extract base name, bracket content, and colon content"""
                base_name = name
                bracket_content = None
                colon_content = None
                
                # Extract content in brackets []
                bracket_match = re.search(r'\[(.*?)\]', str(name))
                if bracket_match:
                    bracket_content = bracket_match.group(1).strip()
                    base_name = re.sub(r'\[.*?\]', '', base_name).strip()
                
                # Extract content after colon :
                if ':' in base_name:
                    parts = base_name.split(':', 1)
                    base_name = parts[0].strip()
                    colon_content = parts[1].strip()
                
                return base_name, bracket_content, colon_content
            
            # Apply parsing
            parsed_data = df[variable_col].apply(parse_variable_name)
            parsed_base = [x[0] for x in parsed_data]
            parsed_bracket = [x[1] for x in parsed_data]
            parsed_colon = [x[2] for x in parsed_data]
            
            # Update variable column with parsed base name
            original_variable_col = variable_col
            df[variable_col] = parsed_base
            
            # Auto-populate Group columns with parsed values
            parsed_values = [parsed_bracket, parsed_colon]
            group_names = ['Categories', 'Runs']  # Names for bracket [] and colon : content
            filled_groups = []
            
            for i, parsed_list in enumerate(parsed_values):
                if any(v is not None for v in parsed_list):
                    # Use specific column names
                    group_col_name = group_names[i]
                    
                    # Check if column exists and is empty, or create new
                    if group_col_name in df.columns:
                        if df[group_col_name].isna().all() or (df[group_col_name].astype(str).str.strip() == '').all():
                            df[group_col_name] = parsed_list
                            filled_groups.append(group_col_name)
                    else:
                        # Create new column
                        df[group_col_name] = parsed_list
                        filled_groups.append(group_col_name)
            
            # Parsing complete - groups auto-populated
        
        with col2:
            # Identify group columns
            remaining_cols = [col for col in all_columns if col != variable_col]
            
            # Include any dynamically created Group columns
            for col in df.columns:
                if col not in remaining_cols and col != variable_col:
                    remaining_cols.append(col)
            
            # Default to selecting columns with 'Group' or 'group' in name, but no limit on quantity
            default_groups = [col for col in remaining_cols if 'Group' in col or 'group' in col or col in ['Categories', 'Runs']]
            
            group_columns = st.multiselect(
                "Select Group Columns (optional)",
                options=remaining_cols,
                default=default_groups,
                help="Select any columns for categorization"
            )
        
        # Remove rows where variable column is empty
        df = df[df[variable_col].notna()]
        df = df[~(df[variable_col].astype(str).str.strip() == '')]
        
        # Time columns are the remaining columns
        used_cols = [variable_col] + group_columns
        time_columns = [col for col in all_columns if col not in used_cols]
        
        # Try to convert time columns to numeric
        try:
            time_values = sorted([float(col) for col in time_columns if col.replace('.','',1).replace('-','',1).isdigit()])
            time_columns_sorted = [str(int(t) if t == int(t) else t) for t in time_values]
        except:
            time_columns_sorted = time_columns
        
        if len(time_columns_sorted) == 0:
            st.error("❌ No time columns detected. Please check your data structure.")
            st.stop()
        
        # Function to parse values with units (B, M, K) and scientific notation
        def parse_value(val):
            """Convert values like '1.34 B', '3.4 M', '1e-06' to numeric"""
            if pd.isna(val):
                return val
            
            if isinstance(val, (int, float)):
                return float(val)
            
            # Convert to string and clean
            val_str = str(val).strip()
            
            # Handle empty strings
            if not val_str or val_str.lower() in ['nan', 'none', '']:
                return None
            
            try:
                # Try direct conversion first (handles scientific notation like 1e-06)
                return float(val_str)
            except ValueError:
                pass
            
            # Handle suffixes (B, M, K, T)
            multipliers = {
                'T': 1e12, 'TRILLION': 1e12,
                'B': 1e9, 'BILLION': 1e9,
                'M': 1e6, 'MILLION': 1e6,
                'K': 1e3, 'THOUSAND': 1e3
            }
            
            val_upper = val_str.upper()
            for suffix, multiplier in multipliers.items():
                if val_upper.endswith(suffix):
                    try:
                        number_part = val_str[:-len(suffix)].strip()
                        return float(number_part) * multiplier
                    except ValueError:
                        pass
            
            # If all else fails, return original
            try:
                return float(val_str)
            except:
                return None
        
        # Apply value parsing to time columns
        for col in time_columns_sorted:
            if col in df.columns:
                df[col] = df[col].apply(parse_value)
        
        # Transform data to long format
        id_vars = [variable_col] + group_columns
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=time_columns_sorted,
            var_name='Time',
            value_name='Value'
        )
        
        # Remove any rows with None/NaN values
        df_long = df_long.dropna(subset=['Value'])
        
        # Convert Time to numeric if possible
        try:
            df_long['Time'] = pd.to_numeric(df_long['Time'])
        except:
            pass
        
        # Filtering options
        st.subheader("🎯 Filter Data")
        
        # Create columns for filters + time range
        filter_cols = st.columns(len(id_vars) + 1)
        filters = {}
        
        # Group filters
        for i, col_name in enumerate(id_vars):
            with filter_cols[i]:
                unique_values = sorted(df_long[col_name].unique().tolist())
                selected = st.multiselect(
                    f"{col_name}",
                    options=unique_values,
                    default=unique_values,
                    key=f"filter_{col_name}"
                )
                if selected:
                    filters[col_name] = selected
        
        # Apply filters
        df_filtered = df_long.copy()
        for col_name, values in filters.items():
            df_filtered = df_filtered[df_filtered[col_name].isin(values)]
        
        # Time range filter (in the last column)
        with filter_cols[len(id_vars)]:
            st.markdown("**Time Range**")
            if df_filtered['Time'].dtype in ['int64', 'float64']:
                min_time = float(df_filtered['Time'].min())
                max_time = float(df_filtered['Time'].max())
                time_range = st.slider(
                    "Years",
                    min_value=min_time,
                    max_value=max_time,
                    value=(min_time, max_time),
                    step=1.0 if (max_time - min_time) > 100 else 0.1,
                    label_visibility="collapsed"
                )
                df_filtered = df_filtered[(df_filtered['Time'] >= time_range[0]) & 
                                          (df_filtered['Time'] <= time_range[1])]
            else:
                selected_times = st.multiselect(
                    "Periods",
                    options=sorted(df_filtered['Time'].unique().tolist()),
                    default=sorted(df_filtered['Time'].unique().tolist()),
                    label_visibility="collapsed"
                )
                df_filtered = df_filtered[df_filtered['Time'].isin(selected_times)]
        
        # Plotting options
        st.subheader("🎨 Visualization Options")
        
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            plot_type = st.selectbox(
                "Plot Type",
                options=["Line Plot", "Scatter Plot", "Bar Chart", "Area Chart"],
                index=0
            )
        
        with viz_col2:
            # Create combined column options
            color_options = [None]
            
            # Add "All Combinations" option first if we have multiple id_vars
            if len(id_vars) > 1:
                color_options.append("All Combinations")
            
            # Add individual columns
            color_options.extend(id_vars)
            
            # Add specific combined options if we have groups
            if len(id_vars) > 1:
                color_options.append(f"{variable_col} + Categories" if "Categories" in id_vars else None)
                if len(id_vars) > 2:
                    color_options.append(f"{variable_col} + Runs" if "Runs" in id_vars else None)
                # Remove None values
                color_options = [opt for opt in color_options if opt is not None]
            
            # Default to "All Combinations" when parsing is enabled
            default_color_index = 0
            if parse_variable and "All Combinations" in color_options:
                default_color_index = color_options.index("All Combinations")
            elif len(color_options) > 1:
                default_color_index = 1
            
            color_by = st.selectbox(
                "Color By",
                options=color_options,
                index=default_color_index,
                key=f"color_by_{parse_variable}",  # Key changes when parse state changes
                help="Column to use for color grouping (can combine Variable with Groups)"
            )
            
            # Create combined columns if needed
            if color_by == "All Combinations":
                # Combine all id_vars in format: Variable[Category] : Run
                def combine_non_null(row):
                    result = ""
                    variable_val = None
                    category_val = None
                    run_val = None
                    other_vals = []
                    
                    for col in id_vars:
                        if pd.notna(row[col]) and str(row[col]).lower() not in ['nan', 'none', '']:
                            if col == variable_col:
                                variable_val = str(row[col])
                            elif col == 'Categories':
                                category_val = str(row[col])
                            elif col == 'Runs':
                                run_val = str(row[col])
                            else:
                                other_vals.append(str(row[col]))
                    
                    # Build the formatted string: Variable[Category] : Run
                    if variable_val:
                        result = variable_val
                    if category_val:
                        result += f"[{category_val}]"
                    if run_val:
                        result += f" : {run_val}"
                    if other_vals:
                        result += " + " + " + ".join(other_vals)
                    
                    return result if result else "Unknown"
                
                df_filtered[color_by] = df_filtered.apply(combine_non_null, axis=1)
            elif color_by and '+' in str(color_by):
                parts = [p.strip() for p in color_by.split('+')]
                # Check if all parts exist in df_filtered
                if all(part in df_filtered.columns for part in parts):
                    def combine_parts(row):
                        values = [str(row[col]) for col in parts if pd.notna(row[col]) and str(row[col]).lower() not in ['nan', 'none', '']]
                        return ' + '.join(values)
                    
                    df_filtered[color_by] = df_filtered.apply(combine_parts, axis=1)
        
        with viz_col3:
            facet_by = st.selectbox(
                "Facet By (optional)",
                options=[None] + id_vars,
                index=0,
                help="Column to create multiple subplots"
            )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                show_markers = st.checkbox("Show markers", value=True if plot_type == "Scatter Plot" else False)
                show_legend = st.checkbox("Show legend", value=True)
            
            with adv_col2:
                log_y = st.checkbox("Log scale Y-axis", value=False)
                connect_gaps = st.checkbox("Connect gaps", value=True)
                
                # Bar chart specific options
                if plot_type == "Bar Chart":
                    bar_mode = st.selectbox(
                        "Bar mode",
                        options=["group", "stack", "overlay"],
                        index=0,
                        help="How to display multiple bars: grouped side-by-side, stacked, or overlaid. Requires Color By to be set."
                    )
                    bin_size = st.number_input(
                        "Time bin size",
                        min_value=1,
                        max_value=100,
                        value=1,
                        step=1,
                        help="Group time periods into bins (e.g., 10 = one bar per 10 years)"
                    )
                    
                    # Warning if bar mode is stack/group but no color grouping
                    if bar_mode in ["stack", "group"] and not color_by:
                        st.warning("⚠️ Stack and group modes require a Color By selection to work properly.")
            
            with adv_col3:
                height = st.number_input("Plot height (px)", min_value=300, max_value=1200, value=450, step=50)
                y_axis_format = st.selectbox(
                    "Y-axis number format",
                    options=["Auto", "Decimal", "Scientific", "Abbreviated (K/M/B)"],
                    index=0,
                    help="How to display values on Y-axis"
                )
        
        # Create plot
        st.subheader("📈 Visualization")
        
        if len(df_filtered) == 0:
            st.warning("⚠️ No data to plot with current filters.")
        else:
            try:
                # Apply time binning for bar charts
                df_plot = df_filtered.copy()  # Work on a copy to avoid modifying original
                if plot_type == "Bar Chart" and 'bin_size' in locals() and bin_size > 1:
                    # Create binned time periods
                    if df_plot['Time'].dtype in ['int64', 'float64']:
                        df_plot['Time_Binned'] = (df_plot['Time'] // bin_size) * bin_size
                        # Aggregate values within bins
                        agg_cols = [col for col in df_plot.columns if col not in ['Time', 'Value', 'Time_Binned']]
                        df_plot = df_plot.groupby(agg_cols + ['Time_Binned'], as_index=False)['Value'].mean()
                        df_plot = df_plot.rename(columns={'Time_Binned': 'Time'})
                
                fig = None
                
                # Prepare common parameters
                plot_params = {
                    'data_frame': df_plot,
                    'x': 'Time',
                    'y': 'Value',
                    'color': color_by,
                    'facet_col': facet_by,
                    'height': height,
                    'log_y': log_y
                }
                
                # Create plot based on type
                if plot_type == "Line Plot":
                    fig = px.line(**plot_params, markers=show_markers)
                elif plot_type == "Scatter Plot":
                    fig = px.scatter(**plot_params)
                elif plot_type == "Bar Chart":
                    fig = px.bar(**plot_params, barmode=bar_mode if 'bar_mode' in locals() else 'group')
                elif plot_type == "Area Chart":
                    fig = px.area(**plot_params)
                
                # Update layout
                if fig:
                    # Configure Y-axis format
                    y_axis_config = {}
                    if y_axis_format == "Scientific":
                        y_axis_config['tickformat'] = '.2e'
                    elif y_axis_format == "Decimal":
                        y_axis_config['tickformat'] = ',.2f'
                    elif y_axis_format == "Abbreviated (K/M/B)":
                        y_axis_config['tickformat'] = '.2s'
                    
                    fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Value",
                        hovermode='x unified',
                        showlegend=show_legend,
                        template='plotly_white',
                        yaxis=y_axis_config
                    )
                    
                    # Update hover template to show full values
                    fig.update_traces(
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:,.2f}<extra></extra>'
                    )
                    
                    # Update traces for line plots
                    if plot_type == "Line Plot" and not show_markers:
                        fig.update_traces(mode='lines')
                    
                    if not connect_gaps:
                        fig.update_traces(connectgaps=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download options
                    st.subheader("💾 Export")
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        # Export filtered data
                        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Filtered Data (CSV)",
                            data=csv_data,
                            file_name="filtered_data.csv",
                            mime="text/csv"
                        )
                    
                    with export_col2:
                        # Export plot as HTML
                        html_data = fig.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="📊 Download Plot (HTML)",
                            data=html_data,
                            file_name="plot.html",
                            mime="text/html"
                        )
                
            except Exception as e:
                st.error(f"❌ Error creating plot: {str(e)}")
                st.exception(e)
        
        # Statistics
        with st.expander("📊 Summary Statistics"):
            if len(df_filtered) > 0:
                st.write("**Basic Statistics:**")
                stats_df = df_filtered.groupby(variable_col)['Value'].describe()
                st.dataframe(stats_df)
                
                if color_by and color_by != variable_col:
                    st.write(f"**Statistics by {color_by}:**")
                    stats_by_group = df_filtered.groupby([color_by, variable_col])['Value'].describe()
                    st.dataframe(stats_by_group)
            else:
                st.info("No data available for statistics.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # Instructions when no data is available
    st.info("👆 Paste your data in the table above or upload a CSV file to get started")
    
    with st.expander("📖 Data Format Guide & Features", expanded=False):
        st.markdown("""
        ### 📝 Expected Data Format
        
        Your CSV file should have the following structure:
        
        | Variable | Group 1 | Group 2 | Group 3 | 2000 | 2001 | 2002 | ... |
        |----------|---------|---------|---------|------|------|------|-----|
        | GDP      | Europe  | High    | Model A | 100  | 105  | 110  | ... |
        | GDP      | Asia    | Medium  | Model A | 80   | 85   | 90   | ... |
        | CO2      | Europe  | High    | Model B | 50   | 52   | 54   | ... |
        
        **Requirements:**
        - One column for **Variable** names
        - Optional: Multiple columns for **grouping/categorization** (e.g., Group 1, Group 2, Group 3)
        - Multiple columns for **time periods** (e.g., years: 2000, 2001, 2002, ...)
        - Numeric values for the time series data
        
        ### ✂️ Variable Name Parsing
        
        You can encode grouping information directly in variable names using `[]` and `:` separators:
        
        **Format:** `BaseVariable[BracketGroup] : ColonGroup`
        
        **Examples:**
        - `GDP[Agriculture] : China` → Variable: GDP, Bracket: Agriculture, Colon: China
        - `CO2[Transport]` → Variable: CO2, Bracket: Transport
        - `Temperature : Europe` → Variable: Temperature, Colon: Europe
        
        Enable "Parse variable names" to automatically extract these groups!
        
        ### ✨ Features
        
        - 📂 **Flexible CSV import** - automatically detects your data structure
        - ✂️ **Variable parsing** - extract groups from variable names using `[]` and `:`
        - 🎯 **Multi-level filtering** - filter by variables and groups
        - 📅 **Time range selection** - focus on specific periods
        - 🎨 **Multiple visualization types** - line, scatter, bar, area charts
        - 🌈 **Color grouping** - differentiate data by categories
        - 📊 **Faceted plots** - create multiple subplots for comparison
        - 💾 **Export capabilities** - download filtered data and interactive plots
        - 📈 **Summary statistics** - quick insights into your data
        """)
