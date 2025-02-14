import streamlit as st
from streamlit_elements import elements, mui, dashboard

# Define a unique identifier for the draggable item
item_id = "markdown_item"

# Define the initial layout of the dashboard
layout = [
    dashboard.Item(item_id, 0, 0, 2, 2),  # (id, x_pos, y_pos, width, height)
]

# Create a button to toggle the display of the draggable window
if st.button("Toggle Markdown Window"):
    st.session_state.show_markdown = not st.session_state.get("show_markdown", False)

# Initialize the elements context
with elements("demo"):
    if st.session_state.get("show_markdown", False):
        # Create a dashboard with draggable and resizable items
        with dashboard.Grid(layout, draggableHandle=".draggable"):
            # Define the draggable and resizable item
            with mui.Card(key=item_id, sx={"padding": 2, "cursor": "move"}):
                # Add a handle for dragging
                mui.Typography("Drag here", className="draggable", variant="h6")
                # Display the Markdown content
                st.markdown(
                    """
                    # Sample Markdown Content

                    - **Bold Text**
                    - *Italic Text*
                    - [Link](https://www.example.com)
                    """
                )
