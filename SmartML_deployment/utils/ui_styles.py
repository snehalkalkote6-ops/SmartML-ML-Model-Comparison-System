"""
SmartML UI Styling & Modern Layout Engine
Injects custom CSS and renders polished cards, metric containers, badges, and headers.
"""

import streamlit as st


def apply_custom_styles():
    """
    Applies custom CSS for a modern, responsive, and polished interface.
    """
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Top Header Styling */
        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
        }
        .main-header h1 {
            color: #FFFFFF !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            margin: 0 0 0.5rem 0 !important;
            letter-spacing: -0.02em;
        }
        .main-header p {
            color: #DBEAFE !important;
            font-size: 1.05rem !important;
            margin: 0 !important;
            max-width: 800px;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease-in-out;
            margin-bottom: 1rem;
        }
        .metric-card:hover {
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08);
            border-color: #CBD5E1;
            transform: translateY(-2px);
        }
        .metric-title {
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .metric-sub {
            color: #94A3B8;
            font-size: 0.8rem;
            margin-top: 0.4rem;
        }

        /* Status Badges */
        .smart-badge {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .badge-blue { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
        .badge-green { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
        .badge-amber { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
        .badge-purple { background: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; }
        .badge-red { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }

        /* Step Card */
        .workflow-step {
            background: #F8FAFC;
            border-left: 4px solid #2563EB;
            padding: 1rem 1.2rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.9rem;
        }
        .workflow-step h4 {
            margin: 0 0 0.3rem 0;
            color: #1E293B;
            font-size: 1rem;
            font-weight: 700;
        }
        .workflow-step p {
            margin: 0;
            color: #64748B;
            font-size: 0.88rem;
            line-height: 1.4;
        }

        /* Info Callout Box */
        .callout-box {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 10px;
            padding: 1.1rem 1.4rem;
            margin: 1.2rem 0;
            color: #166534;
        }

        /* Code/Mono Font */
        code {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.88rem !important;
            background: #F1F5F9 !important;
            padding: 0.15rem 0.4rem !important;
            border-radius: 4px !important;
            color: #0F172A !important;
        }

        /* Footer */
        .footer-note {
            text-align: center;
            padding: 2.5rem 0 1.5rem 0;
            color: #94A3B8;
            font-size: 0.82rem;
            border-top: 1px solid #E2E8F0;
            margin-top: 3.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(title: str, subtitle: str, icon: str = "🤖"):
    """
    Renders top banner header.
    """
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(title: str, value: str, subtext: str = "", badge_text: str = "", badge_color: str = "blue"):
    """
    Renders an HTML metric card.
    """
    badge_html = f'<span class="smart-badge badge-{badge_color}">{badge_text}</span>' if badge_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="metric-title">{title}</div>
                {badge_html}
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_status():
    """
    Renders the persistent sidebar status widget and reset button.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <span style="font-size: 2.4rem;">🤖</span>
                <h3 style="margin: 0.2rem 0 0 0; color: #1E3A8A; font-weight: 800; font-size: 1.4rem;">SmartML</h3>
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 600;">Model Selection & Comparison</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("#### 📌 Current Session Status")

        df_loaded = st.session_state.get("df") is not None
        task = st.session_state.get("task", "None").upper()
        target = st.session_state.get("target_col", "None")
        model_trained = st.session_state.get("is_model_trained", False)
        active_model = st.session_state.get("trained_model_name", "None")

        st.markdown(
            f"""
            <div style="font-size: 0.85rem; line-height: 1.8; color: #334155; margin-bottom: 1rem;">
                • <b>Dataset:</b> <span class="smart-badge {'badge-green' if df_loaded else 'badge-amber'}">{'Loaded' if df_loaded else 'Not Loaded'}</span><br>
                • <b>Target:</b> <code>{target if target else 'Not Selected'}</code><br>
                • <b>Task:</b> <span class="smart-badge badge-blue">{task}</span><br>
                • <b>Trained Model:</b> <span class="smart-badge {'badge-green' if model_trained else 'badge-amber'}">{active_model if model_trained else 'None'}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🔄 Reset All Session State", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #94A3B8; text-align: center; line-height: 1.5;">
                <b>SmartML Platform</b><br>
                Built with Streamlit, Scikit-learn, and Plotly
            </div>
            """,
            unsafe_allow_html=True
        )
