from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from src import config
from src.datasets_manager import DATASETS, download_dataset, load_all, load_dataset_frame, dataset_summary
from src.experiment import run_experiment
from src.ml_models import train_models
from src.statistics import dataset_comparison_test

st.set_page_config(page_title='Text-to-SQL Lab', page_icon='☁️', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''
<style>
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {display:none!important}
.stApp{background:#f6f8fb;color:#172033}.block-container{max-width:1050px;padding:1.2rem 1.4rem 3rem}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:.35rem 0 1rem;border-bottom:1px solid #e6eaf0;margin-bottom:1.2rem}
.brand{font-weight:750;font-size:1.15rem}.brand span{display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;background:#eaf2ff;border-radius:10px;margin-right:.55rem}
.status{font-size:.82rem;background:#ecfdf3;color:#067647;border-radius:999px;padding:.38rem .65rem}
.card{background:white;border:1px solid #e5eaf1;border-radius:15px;padding:1.15rem;margin:.7rem 0;box-shadow:0 5px 18px rgba(16,24,40,.035)}
.hero{padding:2rem .3rem 1rem;text-align:center}.hero h1{font-size:2.2rem;letter-spacing:-.035em;margin:0;color:#101828}.hero p{max-width:650px;margin:.75rem auto;color:#667085;line-height:1.65}
.small{color:#667085;font-size:.9rem}.label{font-weight:700;color:#344054;margin-bottom:.2rem}
[data-testid="stMetric"]{background:white;border:1px solid #e5eaf1;border-radius:13px;padding:.8rem 1rem}
.stButton>button{border-radius:10px;min-height:42px;font-weight:650;background:#175cd3;border:1px solid #175cd3;color:white}
.stButton>button:hover{background:#1849a9;border-color:#1849a9;color:white}
.stTextInput input,.stTextArea textarea,[data-baseweb="select"]>div{border-radius:10px!important;border-color:#d8dee8!important;background:white!important}
[data-testid="stDataFrame"]{border:1px solid #e5eaf1;border-radius:12px;overflow:hidden}
div[role="radiogroup"]{background:white;border:1px solid #e5eaf1;border-radius:12px;padding:.25rem;display:flex;justify-content:center;gap:.2rem}
div[role="radiogroup"] label{padding:.35rem .8rem;border-radius:9px}
@media(max-width:700px){.block-container{padding:1rem}.hero h1{font-size:1.75rem}}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="topbar"><div class="brand"><span>☁️</span>Text-to-SQL Lab</div><div class="status">Research application</div></div>', unsafe_allow_html=True)

page = st.radio('Navigation', ['Ask', 'Datasets', 'Experiment', 'Comparison', 'ML prediction'], horizontal=True, label_visibility='collapsed')

@st.cache_data(show_spinner=False, ttl=3600)
def cached_all():
    return load_all(auto_download=True)


def get_results():
    path = config.OUTPUT_DIR / 'experiment_results.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

if page == 'Ask':
    st.markdown('<div class="hero"><h1>Ask questions. Compare methods.</h1><p>A simple research application for comparing schema-context strategies across BIRD Mini-Dev and Spider.</p></div>', unsafe_allow_html=True)
    frame = cached_all()
    if frame.empty:
        st.error('Datasets could not be loaded. Open the Datasets page and retry the download.')
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        dataset_name = c1.selectbox('Dataset', list(DATASETS))
        subset = frame[frame['dataset'] == dataset_name]
        db_id = c2.selectbox('Database', sorted(subset['db_id'].unique().tolist()))
        examples = subset[subset['db_id'] == db_id].head(100)
        selected = st.selectbox('Benchmark question', examples['question'].tolist())
        record = examples[examples['question'] == selected].iloc[0]
        st.text_area('Question', value=record['question'], height=105, disabled=True)
        st.caption('The benchmark answer is shown for inspection. Live generation is performed from the Experiment page so comparisons remain controlled.')
        if st.button('Show benchmark SQL', use_container_width=True):
            st.session_state['benchmark_sql'] = record['gold_sql']
        if st.session_state.get('benchmark_sql'):
            st.code(st.session_state['benchmark_sql'], language='sql')
        st.markdown('</div>', unsafe_allow_html=True)

elif page == 'Datasets':
    st.header('Datasets')
    st.caption('Both question datasets are downloaded automatically from Hugging Face and cached locally.')
    cols = st.columns(2)
    for col, name in zip(cols, DATASETS):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'### {name}')
            st.write(DATASETS[name]['description'])
            try:
                frame = load_dataset_frame(name, auto_download=False)
                st.caption(f"{len(frame):,} cached questions" if not frame.empty else 'Not downloaded yet')
            except Exception:
                st.caption('Not downloaded yet')
            if st.button(f'Download / refresh {name}', key=f'dl_{name}', use_container_width=True):
                with st.spinner(f'Downloading {name}...'):
                    try:
                        downloaded = download_dataset(name, force=True)
                        cached_all.clear()
                        st.success(f'{len(downloaded):,} questions are ready.')
                    except Exception as exc:
                        st.error(f'Download failed: {exc}')
            st.markdown('</div>', unsafe_allow_html=True)
    all_data = cached_all()
    if not all_data.empty:
        st.subheader('Dataset overview')
        st.dataframe(dataset_summary(all_data).round(2), use_container_width=True, hide_index=True)
        dataset_filter = st.selectbox('Browse', ['Both datasets'] + list(DATASETS))
        view = all_data if dataset_filter == 'Both datasets' else all_data[all_data['dataset'] == dataset_filter]
        search = st.text_input('Search questions')
        if search:
            view = view[view['question'].str.contains(search, case=False, na=False)]
        st.dataframe(view[['dataset','db_id','difficulty','question','gold_sql']].head(500), use_container_width=True, hide_index=True, height=470)

elif page == 'Experiment':
    st.header('Controlled experiment')
    st.caption('Run the same Full, Top-1, Top-3 and Top-5 pattern on one or both datasets.')
    frame = cached_all()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    dataset_choice = st.selectbox('Dataset scope', ['Both datasets'] + list(DATASETS))
    count = st.number_input('Questions per selected scope', min_value=1, max_value=250, value=10)
    provider = st.selectbox('SQL provider', ['OpenAI'])
    st.caption('OpenAI API billing is separate from ChatGPT Plus. API errors are caught and recorded instead of crashing the app.')
    run = st.button('Run experiment', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if run:
        selected = frame if dataset_choice == 'Both datasets' else frame[frame['dataset'] == dataset_choice]
        if dataset_choice == 'Both datasets':
            selected = selected.groupby('dataset', group_keys=False).head(int(count))
        else:
            selected = selected.head(int(count))
        progress = st.progress(0)
        note = st.empty()
        def update(current, total, message):
            progress.progress(current / total)
            note.caption(message)
        with st.spinner('Running controlled comparison...'):
            try:
                output = run_experiment(selected, len(selected), provider, update)
                st.session_state['experiment_summary'] = output['summary']
                st.success('Experiment finished. Results were saved.')
                st.dataframe(output['summary'].round(4), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(str(exc))

elif page == 'Comparison':
    st.header('Dataset comparison')
    results = get_results()
    if results.empty:
        st.info('Run an experiment first. This page will compare BIRD and Spider using the same strategies.')
    else:
        summary_path = config.OUTPUT_DIR / 'dataset_strategy_summary.csv'
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        if not summary.empty:
            st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
        test = dataset_comparison_test(results)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Statistical comparison')
        if test.get('status') == 'ok':
            a,b,c = st.columns(3)
            a.metric('Test', test['test'])
            b.metric('p-value', f"{test['p_value']:.4f}")
            c.metric('Significant', 'Yes' if test['p_value'] < .05 else 'No')
        else:
            st.info(test.get('message'))
        st.markdown('</div>', unsafe_allow_html=True)
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button('Download experiment results', csv, 'experiment_results.csv', 'text/csv')

elif page == 'ML prediction':
    st.header('ML prediction')
    st.caption('Three machine-learning classifiers predict whether a generated SQL query will exactly match the benchmark SQL.')
    results = get_results()
    if results.empty:
        st.info('Run enough experiments first. At least 30 completed rows containing both outcome classes are required.')
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write('Models: Logistic Regression, Random Forest and Gradient Boosting.')
        if st.button('Train and compare models', use_container_width=True):
            with st.spinner('Training models...'):
                try:
                    output = train_models(results)
                    st.session_state['ml_metrics'] = output['metrics']
                    st.session_state['ml_predictions'] = output['predictions']
                except Exception as exc:
                    st.error(str(exc))
        st.markdown('</div>', unsafe_allow_html=True)
        metrics = st.session_state.get('ml_metrics')
        predictions = st.session_state.get('ml_predictions')
        if metrics is not None:
            st.subheader('Model performance')
            st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
            best = metrics.iloc[0]
            st.success(f"Best model: {best['model']} · F1 score {best['f1']:.3f}")
        if predictions is not None:
            with st.expander('Prediction details'):
                st.dataframe(predictions.head(300), use_container_width=True, hide_index=True)
