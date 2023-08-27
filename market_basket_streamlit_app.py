# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:16:28 2023

@author: Haidar
"""


import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import streamlit as st
import plotly.express as px
import hydralit_components as hc
import hydralit as hy


# Define the show_market_basket_analysis function
def show_market_basket_analysis():
    # Set page title
    st.title("Marino Lucci Market Basket Analysis")
    
    # Load the data
    data = pd.read_csv('Data_full_with_Gender.csv')
    
    # Filter sales data
    data_sales = data[data['Invoice Type'] == 'Sales']
    
    # Add a small title to the sidebar
    st.sidebar.markdown("**Market Basket Rules:**")

    # Filter by Client Category
    client_categories = data_sales['Client Category'].unique()
    selected_client_category = st.sidebar.selectbox('Select Client Category', client_categories)
    data_selected = data_sales[data_sales['Client Category'] == selected_client_category]
    
    # Relevant product columns
    product_columns = ['Kind', 'Cut', 'Product Code', 'Product Name', 'Brand', 'Color', 'Size']
    selected_product_columns = st.sidebar.multiselect('Select Product Columns', product_columns)
    
    # Group transactions based on selected product columns
    transaction = data_selected.groupby(['Client', 'Date'])[selected_product_columns].apply(lambda x: frozenset(x.dropna().values.flatten()))
    
    # Input thresholds
    min_support_text = st.sidebar.text_input('Minimum Support (0.001 to 1.000)', '0.010')
    min_support = float(min_support_text)
    
    min_confidence = st.sidebar.slider('Minimum Confidence', 0.0, 1.0, 0.05)
    min_lift = st.sidebar.slider('Minimum Lift', 1.0, 10.0, 2.0)
    min_length = 2  # Always fixed to 2 in your case
    
    
    # Apply Apriori algorithm
    rules = apriori(transaction.tolist(), min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=min_length)
    results = list(rules)
    
    # Create DataFrame from inspection
    def inspect(results):
        antecedent = [tuple(result[2][0][0])[0] for result in results]
        consequent = [tuple(result[2][0][1])[0] for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return list(zip(antecedent, consequent, supports, confidences, lifts))
    
    basket_df = pd.DataFrame(inspect(results), columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    
    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Basket Items based on selected rules:</h3>", unsafe_allow_html=True)
    
    # Display the resulting DataFrame
    st.write(basket_df)
    
    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Visualization:</h3>", unsafe_allow_html=True)
    
    # Slider to select top N rules for plots
    top_n = st.slider("Select Top N Association Rules", 0, 20, 1)
    
    # Sort the DataFrame by 'Lift' column in descending order
    sorted_basket_df_lift = basket_df.sort_values(by='Lift', ascending=False)
    sorted_basket_df_confidence = basket_df.sort_values(by='Confidence', ascending=False)
    sorted_basket_df_support = basket_df.sort_values(by='Support', ascending=False)
    
    # Create columns for buttons
    col1, col2, col3, col4 = st.columns(4)
    
    import matplotlib.pyplot as plt

    # Button to show/hide Plot 1
    show_plot1 = col1.button("Show Items with highest Lift")
    if show_plot1:
        # Plot 1: Top N rules with highest lift
        sorted_lift_df = sorted_basket_df_lift.head(top_n).sort_values(by='Lift', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_lift_df['Antecedent'] + ' -> ' + sorted_lift_df['Consequent'], sorted_lift_df['Lift'])
        ax.set_title(f'Top {top_n} Association Rules with Highest Lift')
        ax.set_xlabel('Lift')
        ax.set_ylabel('Rule')
        st.pyplot(fig)
    
    # Button to show/hide Plot 2
    show_plot2 = col2.button("Show Items with highest Confidence")
    if show_plot2:
        # Plot 2: Top N rules with highest confidence
        sorted_confidence_df = sorted_basket_df_confidence.head(top_n).sort_values(by='Confidence', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_confidence_df['Antecedent'] + ' -> ' + sorted_confidence_df['Consequent'], sorted_confidence_df['Confidence'])
        ax.set_title(f'Top {top_n} Association Rules with Highest Confidence')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Rule')
        st.pyplot(fig)
    
    # Button to show/hide Plot 3
    show_plot3 = col3.button("Show Items with highest Support")
    if show_plot3:
        # Plot 3: Top N rules with highest support
        sorted_support_df = sorted_basket_df_support.head(top_n).sort_values(by='Support', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_support_df['Antecedent'] + ' -> ' + sorted_support_df['Consequent'], sorted_support_df['Support'])
        ax.set_title(f'Top {top_n} Association Rules with Highest Support')
        ax.set_xlabel('Support')
        ax.set_ylabel('Rule')
        st.pyplot(fig)

    
    # # Button to show/hide Plot 4
    # show_plot4 = col4.button("Show Item Links")
    # if show_plot4:
    #     # Plot 4: Enhanced Network Visualization of Association Rules
    #     import networkx as nx
    #     import matplotlib.pyplot as plt
    
    #     # Create a directed graph
    #     G = nx.DiGraph()
    
    #     # Add nodes (items) that have links
    #     items_with_links = set(basket_df['Antecedent']).union(set(basket_df['Consequent']))
    #     for item in items_with_links:
    #         G.add_node(item)
    
    #     # Filter rules based on some criteria (e.g., lift threshold)
    #     filtered_rules = basket_df[basket_df['Lift'] > 5]
    
    #     # Add edges (association rules)
    #     for idx, rule in filtered_rules.iterrows():
    #         G.add_edge(rule['Antecedent'], rule['Consequent'], weight=rule['Lift'])
    
    #     # Set up positions for the nodes
    #     pos = nx.spring_layout(G, seed=42)
    
    #     # Draw nodes with different sizes based on node degree
    #     node_sizes = [5000 * G.degree(node) for node in G.nodes()]
    #     plt.figure(figsize=(20, 20))  # Adjust the figure size here
    #     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
    
    #     # Draw edges with different widths based on edge weights
    #     edge_widths = [0.3 * G[u][v]['weight'] for u, v in G.edges()]
    #     nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
    #     # Draw labels for nodes with adjustments to avoid overlap
    #     node_labels = {node: node[:15] for node in G.nodes()}  # Truncate long labels
    #     nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='black', verticalalignment='baseline')
    
    #     plt.title('Enhanced Network Visualization of Association Rules')
    #     plt.axis('off')
    #     st.pyplot()


# Define the show_recommendation_engine function
def show_recommendation_engine():
    # Set page title
    st.title("Marino Lucci Recommendation Engine")
    
    
    # Sidebar
    st.sidebar.title("Recommendation Engine Tools")

    # Load the data
    data = pd.read_csv(r'C:\Users\Haidar\Desktop\AUB\MSBA\Summer\Capstone\Data\Data_full_with_Gender.csv')

    # Filter sales data
    data_sales = data[data['Invoice Type'] == 'Sales']

    # Filter by Client Category
    client_categories = data_sales['Client Category'].unique()
    selected_client_category = st.sidebar.selectbox('Select Client Category', client_categories)
    data_selected = data_sales[data_sales['Client Category'] == selected_client_category]

    # Input thresholds
    min_support_text = st.sidebar.text_input('Minimum Support (0.001 to 1.000)', '0.010')
    min_support = float(min_support_text)

    min_confidence = st.sidebar.slider('Minimum Confidence', 0.0, 1.0, 0.05)
    min_lift = st.sidebar.slider('Minimum Lift', 1.0, 10.0, 2.0)
    min_length = 2  # Always fixed to 2 in your case

    # Apply Apriori algorithm
    transaction = data_selected.groupby(['Client', 'Date'])['Product Name'].apply(list)
    rules = apriori(transaction.tolist(), min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=min_length)
    results = list(rules)

    # Create DataFrame from inspection
    def inspect(results):
        antecedent = [tuple(result[2][0][0])[0] for result in results]
        consequent = [tuple(result[2][0][1])[0] for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return pd.DataFrame(list(zip(antecedent, consequent, supports, confidences, lifts)), columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])

    basket_df = inspect(results)

    # Multiselect for Product Names
    product_names = data_selected['Product Name'].unique()
    selected_product_names = st.sidebar.multiselect('Select Product Names', product_names)

    # Logic to find recommended complementary products
    if selected_product_names:
        recommended_products_info = []
        for product_name in selected_product_names:
            if product_name in basket_df['Antecedent'].values:
                recommended = basket_df[basket_df['Antecedent'] == product_name].iloc[0]
                complementary_product = recommended['Consequent']
                support = recommended['Support']
                confidence = recommended['Confidence']
                lift = recommended['Lift']
                recommended_products_info.append((complementary_product, support, confidence, lift, 'consequent'))
            elif product_name in basket_df['Consequent'].values:
                recommended = basket_df[basket_df['Consequent'] == product_name].iloc[0]
                complementary_product = recommended['Antecedent']
                support = recommended['Support']
                confidence = recommended['Confidence']
                lift = recommended['Lift']
                recommended_products_info.append((complementary_product, support, confidence, lift, 'antecedent'))
    
        if recommended_products_info:
            st.markdown("<h3 style='font-size: 18px;'>Recommended Complementary Products:</h3>", unsafe_allow_html=True)
            recommended_df = pd.DataFrame(recommended_products_info, columns=['Complementary Product', 'Support', 'Confidence', 'Lift', 'Type'])
            
            # Display the DataFrame
            st.write(recommended_df[['Complementary Product', 'Support', 'Confidence', 'Lift', 'Type']])
            
            # Display explanations
            for index, row in recommended_df.iterrows():
                complementary_product = row['Complementary Product']
                support = row['Support']
                confidence = row['Confidence']
                lift = row['Lift']
                product_type = row['Type']
                
                type_explanation = f"The complementary product ({complementary_product}) is a {product_type} of {product_name}."
                support_explanation = f"The complementary product ({complementary_product}) has a support value of {support:.3f}."
                confidence_explanation = f"The customer is {confidence * 100:.2f}% more likely to purchase {complementary_product} given that {product_name} is purchased."
                lift_explanation = f"The lift value of {lift:.2f} indicates that the customer is {lift:.2f} times more likely to purchase {complementary_product} in addition to {product_name} compared to purchasing {complementary_product} without {product_name}."
                
                st.write(type_explanation)
                st.write(support_explanation)
                st.write(confidence_explanation)
                st.write(lift_explanation)
        else:
            st.warning("No recommendations found for selected products.")

    else:
        st.warning("Select at least one Product Name for recommendations.")
        
    # Button to trigger the search
    search_button = st.button("Search Clients who bought selected products together")
    
    if search_button:
        selected_products = selected_product_names
        clients_bought_together = []
    
        for product_name in selected_products:
            # Find the complementary product's name based on the selected product
            complementary_product_name = recommended_df[(recommended_df['Complementary Product'] == product_name) | (recommended['Antecedent'] == product_name)].iloc[0]['Complementary Product']
    
            # Filter subset data by selected products
            subset_data = data_selected[(data_selected['Product Name'] == product_name) | (data_selected['Product Name'] == complementary_product_name)]
    
            # Group by Client and Date to find transactions where both products were bought together
            grouped_data = subset_data.groupby(['Client', 'Date'])['Product Name'].apply(list)
    
            # Find clients who bought selected products together
            transactions = grouped_data[grouped_data.apply(lambda x: set([product_name, complementary_product_name]).issubset(x))]
            clients_bought_together.extend(transactions.index.get_level_values('Client').tolist())
    
        # Count the occurrences of clients and products bought together
        clients_bought_together_count = pd.Series(clients_bought_together).value_counts()
    
        st.markdown("<h3 style='font-size: 18px;'>Clients who bought selected products together:</h3>", unsafe_allow_html=True)
    
        if not clients_bought_together_count.empty:
            st.write(clients_bought_together_count)
        else:
            st.warning("No clients found who bought selected products together.")

# Add logo image to the sidebar
logo_image = "data:image/jpeg;base64,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"  # Replace with the actual path to your logo image
st.sidebar.image(logo_image, width=75)



# Tab layout
menu_data = [
    {'label': "Market Basket Analysis", 'icon': '📊'},
    {'label': 'Recommendation Engine', 'icon': '💡'},
]

dark_blue_color = 'rgb(0, 0, 128)'  # Change this to your desired dark blue color

# Add custom CSS to adjust the menu bar and page width
st.markdown(
    f"""
    <style>
    .nav-bar {{
        background-color: {dark_blue_color} !important;
        width: 100%;
    }}
    .nav-bar li {{
        width: 50%;
    }}
    .block-container {{
        max-width: 1600px;  /* Adjust the maximum width as needed */
        padding: 2rem;
        margin: 0 auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the custom CSS to adjust the page layout
st.markdown('<div class="block-container">', unsafe_allow_html=True)

selected_menu = hc.nav_bar(
    menu_definition=menu_data,
    override_theme={
        'txc_inactive': 'white',
        'option_active': 'white'
    },
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='sticky'
)

# Display the selected section/page based on the selected menu
if selected_menu == "Market Basket Analysis":
    show_market_basket_analysis()
elif selected_menu == "Recommendation Engine":
    show_recommendation_engine()

# Close the custom block-container
st.markdown('</div>', unsafe_allow_html=True)
# Footer
st.write("© Marino Lucci")

