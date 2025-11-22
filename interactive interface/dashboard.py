import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize App
app = dash.Dash(__name__)

# ==========================================
# 1. Configuration & Styles
# ==========================================
THEME = {
    "bg_color": "#f4f6f9",
    "card_bg": "#ffffff",
    "text_primary": "#2c3e50",
    "text_secondary": "#7f8c8d",
    "accent": "#2980b9",
    "accent_light": "#3498db",
}

styles = {
    "container": {
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "40px",
        "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        "backgroundColor": THEME["bg_color"],
        "minHeight": "100vh",
    },
    "header_container": {
        "textAlign": "center",
        "marginBottom": "40px",
    },
    "header": {
        "color": THEME["text_primary"],
        "fontSize": "36px",
        "fontWeight": "700",
        "marginBottom": "10px",
    },
    "subheader": {
        "color": THEME["text_secondary"],
        "fontSize": "18px",
        "fontWeight": "400",
    },
    "tab_parent": {"marginBottom": "20px"},
    "tab_style": {
        "padding": "15px",
        "fontWeight": "600",
        "color": THEME["text_secondary"],
        "borderBottom": "2px solid #ddd",
        "backgroundColor": "transparent",
    },
    "tab_selected": {
        "borderTop": f'4px solid {THEME["accent"]}',
        "borderBottom": "2px solid transparent",
        "color": THEME["accent"],
        "backgroundColor": "#fff",
        "fontWeight": "bold",
        "padding": "15px",
        "boxShadow": "0 -2px 5px rgba(0,0,0,0.05)",
    },
    "filter_container": {
        "display": "flex",
        "justifyContent": "center",
        "margin": "20px 0 30px 0",
        "padding": "10px",
        "backgroundColor": "#fff",
        "borderRadius": "50px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.03)",
        "width": "fit-content",
        "marginLeft": "auto",
        "marginRight": "auto",
    },
    "image_card": {
        "backgroundColor": THEME["card_bg"],
        "padding": "25px",
        "borderRadius": "12px",
        "boxShadow": "0 4px 15px rgba(0,0,0,0.05)",
        "marginBottom": "30px",
        "textAlign": "center",
        "transition": "transform 0.2s ease",
    },
    "img": {"maxWidth": "100%", "height": "auto", "borderRadius": "6px"},
    "card_title": {
        "marginBottom": "15px",
        "fontSize": "20px",
        "color": THEME["text_primary"],
        "fontWeight": "600",
    },
    "caption": {
        "marginTop": "15px",
        "color": THEME["text_secondary"],
        "fontSize": "15px",
        "lineHeight": "1.6",
    },
}


# Helper: Create Image Card
def create_image_card(filename, title, description=""):
    return html.Div(
        style=styles["image_card"],
        children=[
            html.H3(title, style=styles["card_title"]),
            html.Img(src=app.get_asset_url(filename), style=styles["img"]),
            html.P(description, style=styles["caption"]) if description else None,
        ],
    )


# ==========================================
# 2. App Layout
# ==========================================
app.layout = html.Div(
    style={"backgroundColor": THEME["bg_color"]},
    children=[
        html.Div(
            style=styles["container"],
            children=[
                # Header
                html.Div(
                    style=styles["header_container"],
                    children=[
                        html.H1(
                            "Global AI Job Market Analysis", style=styles["header"]
                        ),
                        html.P(
                            "Visualizing 32,000+ job records: Trends, Skills & Opportunities",
                            style=styles["subheader"],
                        ),
                    ],
                ),
                # Main Navigation (Tabs)
                html.Div(
                    style=styles["tab_parent"],
                    children=[
                        dcc.Tabs(
                            id="main-tabs",
                            value="tab-overview",
                            children=[
                                dcc.Tab(
                                    label="Market Overview",
                                    value="tab-overview",
                                    style=styles["tab_style"],
                                    selected_style=styles["tab_selected"],
                                ),
                                dcc.Tab(
                                    label="Skills & Tools",
                                    value="tab-skills",
                                    style=styles["tab_style"],
                                    selected_style=styles["tab_selected"],
                                ),
                                dcc.Tab(
                                    label="Cluster Analysis",
                                    value="tab-clusters",
                                    style=styles["tab_style"],
                                    selected_style=styles["tab_selected"],
                                ),
                                dcc.Tab(
                                    label="Insights & Opportunities",
                                    value="tab-opportunity",
                                    style=styles["tab_style"],
                                    selected_style=styles["tab_selected"],
                                ),
                            ],
                        )
                    ],
                ),
                # Sub-Filter Control (Dynamic)
                html.Div(
                    id="filter-wrapper",
                    style=styles["filter_container"],
                    children=[
                        dcc.RadioItems(
                            id="sub-filter",
                            options=[],  # Populated by callback
                            value=None,
                            labelStyle={
                                "display": "inline-block",
                                "padding": "10px 20px",
                                "cursor": "pointer",
                                "fontSize": "15px",
                            },
                            inputStyle={"marginRight": "8px"},
                        )
                    ],
                ),
                # Content Area
                html.Div(id="content-area"),
            ],
        )
    ],
)

# ==========================================
# 3. Callbacks (Interaction Logic)
# ==========================================


# Callback 1: Update Sub-filter options based on selected Tab
@app.callback(
    [Output("sub-filter", "options"), Output("sub-filter", "value")],
    [Input("main-tabs", "value")],
)
def update_filter_options(tab):
    if tab == "tab-overview":
        options = [
            {"label": "All Charts", "value": "all"},
            {"label": "Industry & Companies", "value": "industry"},
            {"label": "Hiring Trends", "value": "trends"},
        ]
        return options, "all"

    elif tab == "tab-skills":
        options = [
            {"label": "All Charts", "value": "all"},
            {"label": "Top Rankings", "value": "rankings"},
            {"label": "Heatmaps & Correlations", "value": "heatmaps"},
            {"label": "Trends Over Time", "value": "trends"},
        ]
        return options, "all"

    elif tab == "tab-clusters":
        options = [
            {"label": "All Charts", "value": "all"},
            {"label": "Cluster Profiles (Radar)", "value": "radar"},
            {"label": "Dimensionality Reduction (PCA/t-SNE)", "value": "dr"},
            {"label": "Statistics & Optimization", "value": "stats"},
        ]
        return options, "all"

    elif tab == "tab-opportunity":
        options = [
            {"label": "All Charts", "value": "all"},
            {"label": "Opportunity Index", "value": "index"},
            {"label": "Salary Heatmaps", "value": "salary"},
        ]
        return options, "all"

    return [], None


# Callback 2: Render Images based on Tab AND Sub-filter
@app.callback(
    Output("content-area", "children"),
    [Input("main-tabs", "value"), Input("sub-filter", "value")],
)
def render_content(tab, filter_val):
    if not filter_val:
        return html.Div()  # Prevent initial flash

    content = []

    # --- TAB 1: MARKET OVERVIEW ---
    if tab == "tab-overview":
        if filter_val in ["all", "industry"]:
            content.append(
                create_image_card(
                    "industry_distribution.png",
                    "Industry Distribution",
                    "Automotive and Education sectors lead the demand.",
                )
            )
            content.append(
                create_image_card(
                    "top_10_companies_hiring.png", "Top 10 Hiring Companies"
                )
            )

        if filter_val in ["all", "trends"]:
            content.append(
                create_image_card(
                    "quarterly_job_postings_trend.png",
                    "Quarterly Postings Trend",
                    "Showing the surge in 2024 and contraction in 2025.",
                )
            )
            content.append(
                create_image_card(
                    "job_title_demand_trend.png",
                    "Demand by Job Title",
                    "Data Scientist roles dominate the hiring volume.",
                )
            )

    # --- TAB 2: SKILLS & TOOLS ---
    elif tab == "tab-skills":
        if filter_val in ["all", "rankings"]:
            content.append(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                    },
                    children=[
                        create_image_card(
                            "top_10_skills_distribution.png",
                            "Top 10 Skills",
                            "Python and SQL are the most required skills.",
                        ),
                        create_image_card(
                            "top_10_tools_distribution.png",
                            "Top 10 Tools",
                            "MLflow and LangChain are the preferred tools.",
                        ),
                    ],
                )
            )
            content.append(
                create_image_card(
                    "salary_vs_skills copy.png",
                    "Salary vs. Number of Skills",
                    "Higher skill count correlates with higher potential pay ceiling.",
                )
            )

        if filter_val in ["all", "heatmaps"]:
            content.append(
                create_image_card(
                    "skills_heatmap_by_industry copy.png",
                    "Heatmap: Skills x Industry",
                    "Visualizing skill demand intensity across different sectors.",
                )
            )
            content.append(
                create_image_card(
                    "skills_heatmap_by_role copy.png",
                    "Heatmap: Skills x Job Role",
                    "Mapping core competencies to specific job titles.",
                )
            )

        if filter_val in ["all", "trends"]:
            content.append(
                create_image_card(
                    "skill_demand_trend.png",
                    "Skill Demand Trends",
                    "Longitudinal tracking of key technologies.",
                )
            )

    # --- TAB 3: CLUSTERS ---
    elif tab == "tab-clusters":
        if filter_val in ["all", "radar"]:
            content.append(
                create_image_card(
                    "cluster_radar_charts.png",
                    "Cluster Profiles (Radar Charts)",
                    "Comparing Salary, Skill Count, and Job Volume across the 3 identified clusters.",
                )
            )
            content.append(
                create_image_card(
                    "cluster_distribution_analysis.png", "Cluster Distribution Stats"
                )
            )

        if filter_val in ["all", "dr"]:
            content.append(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                    },
                    children=[
                        create_image_card(
                            "clusters_pca_FIXED.png", "PCA Projection (2D)"
                        ),
                        create_image_card(
                            "clusters_tsne.png", "t-SNE Projection (Non-Linear)"
                        ),
                    ],
                )
            )
            content.append(
                create_image_card("clusters_3d_pca.png", "3D PCA Visualization")
            )

        if filter_val in ["all", "stats"]:
            content.append(
                create_image_card(
                    "hierarchical_dendrogram.png", "Hierarchical Clustering Dendrogram"
                )
            )
            content.append(
                create_image_card(
                    "cluster_optimization.png",
                    "K-Means Optimization (Elbow & Silhouette)",
                )
            )

    # --- TAB 4: OPPORTUNITIES ---
    elif tab == "tab-opportunity":
        if filter_val in ["all", "index"]:
            content.append(
                create_image_card(
                    "opportunity_index_ranking.png",
                    "AI Job Opportunity Index",
                    "Ranking roles based on a composite score of salary, demand, and competition.",
                )
            )
            content.append(
                create_image_card(
                    "opportunity_components_heatmap.png", "Opportunity Index Components"
                )
            )

        if filter_val in ["all", "salary"]:
            content.append(
                create_image_card(
                    "salary_heatmap_by_region_industry.png",
                    "Global Salary Heatmap (Region vs Industry)",
                )
            )

    return html.Div(content)


if __name__ == "__main__":
    app.run(debug=True)
