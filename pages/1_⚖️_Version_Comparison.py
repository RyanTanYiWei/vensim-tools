import streamlit as st
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

st.set_page_config(layout="wide", page_title="Version Comparisons", page_icon="⚖️")

# Path to CSS file
css_file = Path(__file__).parent.parent / "assets" / "style.css"

if css_file.exists():
    css_content = css_file.read_text()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

st.title("⚖️ Version Comparisons")
st.caption("Built for Modelling Teams Who Struggle With Version Control")
st.warning("⚠️ This tool assumes two model files represent different **versions** of the same model.")
st.markdown(
    """
    Upload two `.mdl` files to compare. The tool will show:  
    🔹 **Changed formulas** for variables present in both models  
    🔹 **New variables** in Model A  
    🔹 **New variables** in Model B  
    """
    )

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("First Vensim model (.mdl)", type=["mdl"], key="file_a")
with col2:
    file_b = st.file_uploader("Second Vensim model (.mdl)", type=["mdl"], key="file_b")


def save_to_temp(uploaded_file) -> Path:
    if not uploaded_file:
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="vensim_"))
    tmp_path = tmp_dir / Path(uploaded_file.name).name
    tmp_path.write_bytes(uploaded_file.read())
    return tmp_path


def extract_equations_with_pysd(model_path: Path, parse_views: bool = True) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """
    Returns: (equations, debug, var_to_view) where var_to_view maps variable name to view/section name
    Args:
        model_path: Path to the .mdl file
        parse_views: If False, skips view parsing for faster performance
    """
    try:
        import pysd
    except Exception as e:
        return {}, [f"PySD import failed: {e}"], {}

    equations: Dict[str, str] = {}
    var_to_view: Dict[str, str] = {}  # Maps variable name to its view/section
    debug: List[str] = [f"Using PySD {getattr(pysd, '__version__', 'unknown')} for '{model_path.name}'"]

    # First, extract equations using raw parse
    try:
        from pysd.translators.vensim.vensim_file import VensimFile  # type: ignore
        vf = VensimFile(str(model_path))
        vf.parse()
        sections = getattr(vf, "sections", []) or []
        debug.append(f"Raw parse sections: {len(sections)}")

        def _stringify_equation(obj):
            for attr in [
                "rhs", "equation", "expr", "expression", "raw_equation", "raw", "text", "value", "definition",
            ]:
                val = getattr(obj, attr, None)
                if val is not None:
                    return str(val)
            for method in ["to_vensim", "to_text", "to_string"]:
                fn = getattr(obj, method, None)
                if callable(fn):
                    try:
                        return str(fn())
                    except Exception:
                        pass
            try:
                return repr(obj)
            except Exception:
                return None

        for section in sections:
            elements = getattr(section, "elements", []) or []
            for element in elements:
                name = (
                    getattr(element, "name", None)
                    or getattr(element, "canon_name", None)
                    or getattr(getattr(element, "component", None) or object(), "name", None)
                )
                equation_text = _stringify_equation(element)
                component = getattr(element, "component", None)
                if (not equation_text or len(equation_text) < 2) and component is not None:
                    eq2 = _stringify_equation(component)
                    if eq2 and len(eq2) > 1:
                        equation_text = eq2
                if name and equation_text:
                    equations[str(name)] = str(equation_text).strip()
        debug.append(f"Raw parse equations extracted: {len(equations)}")
    except Exception as e:
        debug.append(f"Raw parse failed: {e}")

    # Now extract view information using PySD's split_views feature (only if requested)
    if parse_views:
        try:
            import re
            import tempfile
            
            # Read and clean the .mdl text in memory
            text = model_path.read_text(encoding='utf-8', errors='ignore')
            cleaned_text = re.sub(r"\(\d+,\d+\)\|", "", text)
            
            # Create a temporary file for PySD
            tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".mdl", delete=False, encoding='utf-8')
            tmp_file.write(cleaned_text)
            tmp_file.flush()
            tmp_file.close()
            
            # Parse with split_views
            models = pysd.read_vensim(tmp_file.name, split_views=True, initialize=False)
            
            # Map PySD variable names back to Vensim names
            py_to_vensim = {py_name: vensim_name for vensim_name, py_name in models.namespace.items()}
            
            # Build the view mapping
            for view_name, py_vars in getattr(models, "modules", {}).items():
                for py_var in py_vars:
                    vensim_name = py_to_vensim.get(py_var, py_var)
                    var_to_view[vensim_name] = view_name
            
            debug.append(f"Views parsed: {len(var_to_view)} variables across {len(getattr(models,'modules',{}))} views")
            
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_file.name)
            except:
                pass
                
        except Exception as e:
            debug.append(f"PySD view parsing failed: {e}")
    else:
        debug.append("View parsing skipped (fast mode)")

    return equations, debug, var_to_view


def diff_equation_maps(a: Dict[str, str], b: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]], Dict[str, str]]:
    only_in_a = {k: a[k] for k in a.keys() - b.keys()}
    only_in_b = {k: b[k] for k in b.keys() - a.keys()}
    changed = {}
    for k in a.keys() & b.keys():
        if (a.get(k) or "").strip() != (b.get(k) or "").strip():
            changed[k] = (a.get(k, ""), b.get(k, ""))
    return only_in_a, changed, only_in_b


if file_a and file_b:
    st.markdown("### Comparison Mode")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        compare_full = st.button("⚡ Compare Full Models (Fast)", use_container_width=True)
    with col_btn2:
        compare_views = st.button("🔍 Compare Model Views (Slow)", use_container_width=True)
    
    st.info("**Fast Mode**: Compares all variables without view information. **Slow Mode**: Parses view/module information for filtering.")
    
    if compare_full or compare_views:
        parse_views = compare_views  # Only parse views in slow mode
        with st.spinner("Parsing models with PySD..."):
            path_a = save_to_temp(file_a)
            path_b = save_to_temp(file_b)
            eq_a, dbg_a, views_a = extract_equations_with_pysd(path_a, parse_views=parse_views)
            eq_b, dbg_b, views_b = extract_equations_with_pysd(path_b, parse_views=parse_views)
        
        st.session_state["_diff_result"] = {
            "eq_a": eq_a,
            "eq_b": eq_b,
            "views_a": views_a,
            "views_b": views_b,
            "dbg_a": dbg_a,
            "dbg_b": dbg_b,
            "a_name": Path(file_a.name).name,
            "b_name": Path(file_b.name).name,
            "parse_views": parse_views,
        }

    diff = st.session_state.get("_diff_result")
    if diff:
        eq_a = diff["eq_a"]; eq_b = diff["eq_b"]
        dbg_a = diff["dbg_a"]; dbg_b = diff["dbg_b"]
        a_name = diff["a_name"]; b_name = diff["b_name"]
        views_a = diff.get("views_a", {}); views_b = diff.get("views_b", {})

        if not eq_a or not eq_b:
            st.warning("Unable to extract equations from one or both models.")
            with st.expander("Diagnostics: Model A"):
                for line in dbg_a:
                    st.write(line)
                st.write(f"Equations extracted: {len(eq_a)}")
            with st.expander("Diagnostics: Model B"):
                for line in dbg_b:
                    st.write(line)
                st.write(f"Equations extracted: {len(eq_b)}")
        else:
            # Check if views were parsed
            parse_views = diff.get("parse_views", False)
            
            if parse_views:
                # --- Filter Model Views Section ---
                st.subheader("🔍 Filter Model Views")
                st.markdown("Select which views/modules to include in the comparison. By default, all views are included.")
                
                # Get all unique views from both models
                all_views = sorted(set(list(views_a.values()) + list(views_b.values())))
                
                if all_views:
                    selected_views = st.multiselect(
                        "Select views to include",
                        options=all_views,
                        default=all_views,
                        help="Uncheck views you want to exclude from the comparison"
                    )
                    
                    # Filter equations based on selected views
                    if selected_views:
                        eq_a_filtered = {k: v for k, v in eq_a.items() if views_a.get(k, "Unknown View") in selected_views}
                        eq_b_filtered = {k: v for k, v in eq_b.items() if views_b.get(k, "Unknown View") in selected_views}
                    else:
                        eq_a_filtered = {}
                        eq_b_filtered = {}
                    
                    st.info(f"📊 Filtered: {len(eq_a_filtered)}/{len(eq_a)} variables from Model A, {len(eq_b_filtered)}/{len(eq_b)} variables from Model B")
                else:
                    st.info("No view information available. Showing all variables.")
                    eq_a_filtered = eq_a
                    eq_b_filtered = eq_b
                
                st.markdown("---")
            else:
                # Fast mode: no filtering
                eq_a_filtered = eq_a
                eq_b_filtered = eq_b
            
            # Use filtered equations for comparison
            only_a, changed, only_b = diff_equation_maps(eq_a_filtered, eq_b_filtered)

            total_a = len(eq_a_filtered)
            total_b = len(eq_b_filtered)
            common = set(eq_a_filtered.keys()) & set(eq_b_filtered.keys())
            changed_count = len(changed)
            similar_count = max(len(common) - changed_count, 0)

            # --- Unified Summary ---
            st.subheader("Summary")
            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)

            c1.metric(label=f"Variables in {a_name}", value=total_a)
            c2.metric(label=f"Variables in {b_name}", value=total_b)
            c3.metric(label="Common", value=len(common))
            c4.metric(label="Similar", value=similar_count)
            c5.metric(label="Changed", value=changed_count)
            c6.metric(label=f"Only in {a_name}/{b_name}", value=f"{len(only_a)} / {len(only_b)}")

            # --- Render differences ---
            def _render_truncated(label: str, text: str, max_chars: int = 200):
                if text is None:
                    text = ""
                text = str(text)
                truncated = text if len(text) <= max_chars else text[:max_chars] + " …"
                st.code(text) # no truncation

            def render_changed():
                if not changed:
                    st.info("No changed formulas.")
                    return
                with st.expander("Changed formulas (expand/collapse)", expanded=True):
                    for name, (ea, eb) in sorted(changed.items()):
                        st.markdown(f"<div style='font-weight:600; font-size:0.95rem; margin-bottom:0.5rem;'>{name}</div>", unsafe_allow_html=True)
                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.markdown(f"<div style='font-weight:500; opacity:0.7; margin-bottom:0.25rem;'>{a_name} formula:</div>", unsafe_allow_html=True)
                            _render_truncated("", ea, max_chars=400)

                        with col_right:
                            st.markdown(f"<div style='font-weight:500; opacity:0.7; margin-bottom:0.25rem;'>{b_name} formula:</div>", unsafe_allow_html=True)
                            _render_truncated("", eb, max_chars=400)

                        st.markdown("<hr style='border:none; border-top:1px solid rgba(0,0,0,0.08); margin:0.5rem 0;'>", unsafe_allow_html=True)


            def render_only_a():
                if not only_a:
                    st.info(f"No variables only in {a_name}.")
                    return
                with st.expander(f"Only in {a_name} (expand/collapse)", expanded=True):
                    for name, eq in sorted(only_a.items()):
                        st.markdown(f"- **{name}**")
                        _render_truncated(f"{a_name} formula", eq)

            def render_only_b():
                if not only_b:
                    st.info(f"No variables only in {b_name}.")
                    return
                with st.expander(f"Only in {b_name} (expand/collapse)", expanded=True):
                    for name, eq in sorted(only_b.items()):
                        st.markdown(f"- **{name}**")
                        _render_truncated(f"{b_name} formula", eq)

            render_changed()
            render_only_a()
            render_only_b()

            # CSV export
            import io, csv
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Type", "Name", a_name + " Equation", b_name + " Equation"])
            for name, (ea, eb) in changed.items():
                writer.writerow(["changed", name, ea, eb])
            for name, ea in only_a.items():
                writer.writerow(["only_in_a", name, ea, ""]) 
            for name, eb in only_b.items():
                writer.writerow(["only_in_b", name, "", eb])
            st.download_button(
                label="Download differences (CSV)",
                data=output.getvalue(),
                file_name=f"vensim_formula_diff_{a_name}_vs_{b_name}.csv",
                mime="text/csv",
            )
