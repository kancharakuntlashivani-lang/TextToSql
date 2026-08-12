from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from src import config
from src.datasets_manager import (
    DATASETS,
    download_dataset,
    load_all,
    load_dataset_frame,
    dataset_summary,
    register_uploaded_dataset,
    list_uploaded_datasets,
    all_dataset_names,
    inspect_sqlite_database,
    get_uploaded_metadata,
    uploaded_sqlite_path,
)
from src.experiment import run_experiment
from src.ml_models import train_models
from src.statistics import (
    dataset_comparison_test,
    all_strategy_tests,
    strategy_efficiency_test,
    complete_statistical_analysis,
)
from src import core
from src.core import (
    list_databases,
    run_strategy,
    run_custom_question,
    migrate_uploaded_sqlite_to_postgres,
    normalise_dataset,
)

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
    st.markdown('<div class="hero"><h1>Ask a database question</h1><p>Enter your own question or select a benchmark question. The application generates PostgreSQL, executes it, and shows whether it matches the expected result when ground truth is available.</p></div>', unsafe_allow_html=True)

    if not core.database_health():
        st.error('PostgreSQL connection failed.')

        if getattr(core, 'DATABASE_HEALTH_ERROR', ''):
            st.code(
                core.DATABASE_HEALTH_ERROR,
                language='text',
            )

        st.info(
            'Check DATABASE_URL in the Render web-service environment. '
            'Use the exact Internal Database URL from the Render PostgreSQL service. '
            'Also confirm that the web service and PostgreSQL database use the same region.'
        )

        st.stop()

    frame = cached_all()
    if frame.empty:
        st.error('Datasets could not be loaded. Open the Datasets page and retry the download.')
        st.stop()

    display_to_key = {
        'BIRD Mini-Dev': 'bird',
        'Spider': 'spider',
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])

    dataset_choices = all_dataset_names()
    dataset_name = c1.selectbox('Dataset', dataset_choices)

    is_uploaded_dataset = dataset_name not in display_to_key

    if is_uploaded_dataset:
        dataset_key = normalise_dataset(dataset_name)
    else:
        dataset_key = display_to_key[dataset_name]

    available_databases = list_databases(dataset_key)

    if not available_databases:
        if is_uploaded_dataset:
            metadata = get_uploaded_metadata(dataset_name)

            st.warning(
                'This uploaded dataset has not been migrated to PostgreSQL yet.'
            )

            if metadata:
                st.caption(
                    'Open the Datasets page, migrate the uploaded SQLite database, '
                    'then return here to ask questions.'
                )
        else:
            st.error(
                f'No migrated PostgreSQL schemas were found for {dataset_name}.'
            )

        st.stop()

    subset = frame[frame['dataset'] == dataset_name].copy()
    benchmark_databases = set(subset['db_id'].astype(str))
    database_options = [db for db in available_databases if db in benchmark_databases]
    if not database_options:
        database_options = available_databases

    db_id = c2.selectbox('Database', sorted(database_options))
    strategy_label = c3.selectbox(
        'Schema strategy',
        ['Relevant schema (Top-5)', 'Full schema', 'Top-1', 'Top-3'],
    )
    strategy_map = {
        'Relevant schema (Top-5)': 'top_5',
        'Full schema': 'full',
        'Top-1': 'top_1',
        'Top-3': 'top_3',
    }

    mode_options = ['Enter my own question']

    if not is_uploaded_dataset:
        mode_options.append('Use benchmark question')
    elif not subset.empty:
        mode_options.append('Use benchmark question')

    mode = st.radio(
        'Question mode',
        mode_options,
        horizontal=True,
    )

    gold_sql = ''
    evidence = ''
    benchmark_record = None

    if mode == 'Use benchmark question':
        examples = subset[
            subset['db_id'].astype(str) == str(db_id)
        ].head(200)

        if examples.empty:
            st.warning(
                'No benchmark questions were found for this migrated database. '
                'Use your own question instead.'
            )

            question = st.text_area(
                'Question',
                height=110,
                placeholder='Example: Show the five customers with the highest total payments.',
            )

        else:
            selected = st.selectbox(
                'Benchmark question',
                examples['question'].tolist(),
            )

            benchmark_record = (
                examples[
                    examples['question'] == selected
                ].iloc[0]
            )

            question = st.text_area(
                'Question',
                value=str(
                    benchmark_record['question']
                ),
                height=110,
            )

            gold_sql = str(
                benchmark_record.get(
                    'gold_sql',
                    '',
                )
                or ''
            )

            evidence = str(
                benchmark_record.get(
                    'evidence',
                    '',
                )
                or ''
            ).strip()

            if gold_sql.strip():
                st.caption(
                    'Ground-truth SQL is available, so the application can '
                    'show Correct or Incorrect.'
                )
            else:
                st.caption(
                    'This benchmark row does not contain verified SQL, so only '
                    'execution success can be shown.'
                )

            if evidence:
                with st.expander(
                    'Benchmark evidence / hint'
                ):
                    st.write(
                        evidence
                    )

    else:
        question = st.text_area(
            'Enter your natural-language question',
            height=110,
            placeholder='Example: Show the five customers with the highest total payments.',
        )

        with st.expander(
            'Optional: provide expected SQL to verify correctness'
        ):
            gold_sql = st.text_area(
                'Expected SQL',
                height=120,
                placeholder='Leave empty when the correct answer is not known.',
            )

            st.caption(
                'Without expected SQL, the application can show SQL validity, '
                'execution success and database results, but it cannot claim '
                '100% correctness.'
            )

    generate = st.button('Generate and run SQL', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if generate:
        if not question.strip():
            st.warning('Enter a question first.')
        else:
            with st.spinner('Retrieving schema, generating SQL and executing the query...'):
                try:
                    if (
                        mode == 'Enter my own question'
                        and not gold_sql.strip()
                    ):
                        result = run_custom_question(
                            question=question.strip(),
                            dataset=dataset_key,
                            db_id=db_id,
                            strategy=strategy_map[strategy_label],
                            max_repair_attempts=1,
                        )
                    else:
                        result = run_strategy(
                            question=question.strip(),
                            db_id=db_id,
                            strategy=strategy_map[strategy_label],
                            gold_sql=gold_sql.strip() or None,
                            dataset=dataset_key,
                            evidence=evidence or None,
                            max_repair_attempts=1,
                        )

                    st.session_state['ask_result'] = result
                except Exception as exc:
                    st.error(f'Generation failed: {exc}')

    result = st.session_state.get('ask_result')
    if result and result.get('question') == question.strip() and result.get('db_id') == db_id:
        st.subheader('Generated SQL')
        st.code(result['generated_sql'], language='sql')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('SQL valid / executed', 'Yes' if result['success'] else 'No')
        m2.metric('Generation time', f"{result['generation_latency']:.2f} s")
        m3.metric('Execution time', f"{result['execution_time']:.3f} s")
        m4.metric('Prompt tokens', int(result['prompt_tokens']))

        if result.get('retrieved_tables'):
            with st.expander('Schema tables supplied to the model'):
                st.write(result['retrieved_tables'])

        if result.get('evidence'):
            with st.expander('Evidence supplied to the model'):
                st.write(result['evidence'])

        if result.get('repair_attempts', 0):
            st.info(
                f"Automatic SQL repair attempts: {result['repair_attempts']}"
            )

            with st.expander('SQL repair details'):
                st.json(
                    result.get(
                        'repair_history',
                        [],
                    )
                )

        if result['success']:
            st.success('The SQL is valid and executed successfully. Execution success does not by itself prove the answer is correct.')
            if result.get('rows'):
                st.dataframe(
                    pd.DataFrame(result['rows'], columns=result['columns']),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info('The query executed successfully but returned no rows.')
        else:
            st.error(result.get('error') or 'The generated SQL could not be executed.')

        if gold_sql.strip():
            st.subheader('Correctness check')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Expected SQL**')
                st.code(gold_sql, language='sql')
            with c2:
                st.markdown('**Evaluation**')
                execution_correct = result.get('execution_accuracy') == 1
                exact_correct = result.get('exact_match') == 1
                if execution_correct:
                    st.success('TRUE — generated SQL returned the expected result.')
                else:
                    st.error('FALSE — generated SQL did not return the expected result.')
                st.write(f"Exact SQL match: {'Yes' if exact_correct else 'No'}")
                st.write(f"Execution-result match: {'Yes' if execution_correct else 'No'}")
        else:
            st.info('Accuracy is not available for this custom question because no verified expected SQL was supplied.')

elif page == 'Datasets':
    st.header('Datasets')
    st.caption(
        'Use the built-in BIRD and Spider benchmarks or register your own '
        'Text-to-SQL dataset for later PostgreSQL migration and querying.'
    )

    st.subheader('Built-in benchmarks')

    cols = st.columns(2)

    for col, name in zip(cols, DATASETS):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'### {name}')
            st.write(DATASETS[name]['description'])

            try:
                frame = load_dataset_frame(
                    name,
                    auto_download=False,
                )

                st.caption(
                    f"{len(frame):,} cached questions"
                    if not frame.empty
                    else 'Not downloaded yet'
                )

            except Exception:
                st.caption('Not downloaded yet')

            if st.button(
                f'Download / refresh {name}',
                key=f'dl_{name}',
                use_container_width=True,
            ):
                with st.spinner(
                    f'Downloading {name}...'
                ):
                    try:
                        downloaded = download_dataset(
                            name,
                            force=True,
                        )

                        cached_all.clear()

                        st.success(
                            f'{len(downloaded):,} questions are ready.'
                        )

                    except Exception as exc:
                        st.error(
                            f'Download failed: {exc}'
                        )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader('Upload your own dataset')
    st.caption(
        'Register a dataset using a questions file and/or one SQLite database. '
        'Question files can be CSV, JSON or JSONL. SQLite files can be '
        '.sqlite, .sqlite3 or .db.'
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    upload_name = st.text_input(
        'Dataset name',
        placeholder='Example: My Healthcare Database',
    )

    upload_description = st.text_area(
        'Description',
        height=80,
        placeholder='Short description of the uploaded dataset.',
    )

    q_col, db_col = st.columns(2)

    questions_upload = q_col.file_uploader(
        'Questions file',
        type=[
            'csv',
            'json',
            'jsonl',
        ],
        help=(
            'Recommended columns/fields: question, db_id, gold_sql or query. '
            'Evidence and difficulty are optional.'
        ),
    )

    sqlite_upload = db_col.file_uploader(
        'SQLite database',
        type=[
            'sqlite',
            'sqlite3',
            'db',
        ],
        help=(
            'Upload one SQLite database. After registration, use the migration '
            'button below to move it into PostgreSQL.'
        ),
    )

    if st.button(
        'Register uploaded dataset',
        use_container_width=True,
    ):
        if not upload_name.strip():
            st.warning(
                'Enter a dataset name.'
            )

        elif (
            questions_upload is None
            and sqlite_upload is None
        ):
            st.warning(
                'Upload at least a questions file or an SQLite database.'
            )

        else:
            try:
                upload_temp_dir = (
                    config.DATA_DIR
                    / '_upload_temp'
                )

                upload_temp_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                questions_temp_path = None
                sqlite_temp_path = None

                if questions_upload is not None:
                    questions_temp_path = (
                        upload_temp_dir
                        / questions_upload.name
                    )

                    questions_temp_path.write_bytes(
                        questions_upload.getbuffer()
                    )

                if sqlite_upload is not None:
                    sqlite_temp_path = (
                        upload_temp_dir
                        / sqlite_upload.name
                    )

                    sqlite_temp_path.write_bytes(
                        sqlite_upload.getbuffer()
                    )

                metadata = register_uploaded_dataset(
                    dataset_name=upload_name.strip(),
                    questions_file=questions_temp_path,
                    sqlite_file=sqlite_temp_path,
                    description=upload_description.strip(),
                )

                cached_all.clear()

                st.success(
                    f"Dataset '{metadata['dataset_name']}' was registered successfully."
                )

                if questions_temp_path and questions_temp_path.exists():
                    try:
                        questions_temp_path.unlink()
                    except Exception:
                        pass

                if sqlite_temp_path and sqlite_temp_path.exists():
                    try:
                        sqlite_temp_path.unlink()
                    except Exception:
                        pass

            except Exception as exc:
                st.error(
                    f'Upload failed: {exc}'
                )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_names = list_uploaded_datasets()

    if uploaded_names:
        st.subheader('Registered uploaded datasets')

        for uploaded_name in uploaded_names:
            metadata = get_uploaded_metadata(
                uploaded_name
            )

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'### {uploaded_name}'
            )

            st.write(
                metadata.get(
                    'description',
                    'User uploaded Text-to-SQL dataset.',
                )
            )

            details = {
                'Questions file': (
                    metadata.get(
                        'questions_file'
                    )
                    or 'Not provided'
                ),
                'SQLite file': (
                    metadata.get(
                        'sqlite_file'
                    )
                    or 'Not provided'
                ),
            }

            st.json(details)

            sqlite_path = metadata.get(
                'sqlite_file'
            )

            if sqlite_path:
                action_col1, action_col2 = st.columns(2)

                if action_col1.button(
                    f'Inspect schema: {uploaded_name}',
                    key=f'inspect_{uploaded_name}',
                    use_container_width=True,
                ):
                    try:
                        schema = inspect_sqlite_database(
                            sqlite_path
                        )

                        if not schema:
                            st.info(
                                'No user tables were found in the SQLite database.'
                            )

                        else:
                            st.success(
                                f'{len(schema)} tables found.'
                            )

                            for table_name, columns in schema.items():
                                with st.expander(
                                    table_name
                                ):
                                    st.dataframe(
                                        pd.DataFrame(
                                            columns
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                    except Exception as exc:
                        st.error(
                            f'Schema inspection failed: {exc}'
                        )

                if action_col2.button(
                    f'Migrate to PostgreSQL: {uploaded_name}',
                    key=f'migrate_{uploaded_name}',
                    use_container_width=True,
                ):
                    with st.spinner(
                        'Migrating SQLite tables into PostgreSQL...'
                    ):
                        try:
                            source_path = uploaded_sqlite_path(
                                uploaded_name
                            )

                            if source_path is None:
                                raise FileNotFoundError(
                                    'Uploaded SQLite database is not available.'
                                )

                            migration = migrate_uploaded_sqlite_to_postgres(
                                dataset=uploaded_name,
                                sqlite_path=source_path,
                                db_id=source_path.stem,
                                replace=True,
                            )

                            core.retriever.clear()

                            st.success(
                                'Migration completed successfully.'
                            )

                            m1, m2, m3 = st.columns(3)

                            m1.metric(
                                'PostgreSQL schema',
                                migration['schema'],
                            )

                            m2.metric(
                                'Tables migrated',
                                migration['table_count'],
                            )

                            m3.metric(
                                'Rows migrated',
                                migration['row_count'],
                            )

                            st.dataframe(
                                pd.DataFrame(
                                    migration['tables']
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.caption(
                                'The uploaded database is now available on the Ask page.'
                            )

                        except Exception as exc:
                            st.error(
                                f'Migration failed: {exc}'
                            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

    all_data = cached_all()

    if not all_data.empty:
        st.subheader(
            'Dataset overview'
        )

        st.dataframe(
            dataset_summary(
                all_data
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )

        browse_options = [
            'All datasets'
        ] + all_dataset_names()

        dataset_filter = st.selectbox(
            'Browse',
            browse_options,
        )

        if dataset_filter == 'All datasets':
            view = all_data

        else:
            view = all_data[
                all_data[
                    'dataset'
                ]
                == dataset_filter
            ]

        search = st.text_input(
            'Search questions'
        )

        if search:
            view = view[
                view[
                    'question'
                ].str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]

        display_columns = [
            column
            for column in [
                'dataset',
                'db_id',
                'difficulty',
                'question',
                'gold_sql',
                'evidence',
            ]
            if column in view.columns
        ]

        st.dataframe(
            view[
                display_columns
            ].head(500),
            use_container_width=True,
            hide_index=True,
            height=470,
        )


elif page == 'Experiment':
    st.header('Controlled experiment')
    st.caption(
        'Run the same benchmark questions through Full Schema, Top-1, '
        'Top-3 and Top-5 using the real PostgreSQL schema and compare '
        'execution accuracy, SQL validity, latency and token usage.'
    )

    frame = cached_all()

    if frame.empty:
        st.info(
            'No benchmark questions are available. Download BIRD or Spider first.'
        )
        st.stop()

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    dataset_choice = st.selectbox(
        'Dataset scope',
        [
            'Both datasets',
            'BIRD Mini-Dev',
            'Spider',
        ],
    )

    count = st.number_input(
        'Questions per selected dataset',
        min_value=1,
        max_value=250,
        value=5,
        step=1,
    )

    provider = st.selectbox(
        'SQL provider',
        ['OpenAI'],
    )

    st.caption(
        'Each benchmark question is evaluated under four schema strategies. '
        'Start with 3–5 questions to validate the pipeline before running a '
        'larger experiment.'
    )

    run = st.button(
        'Run experiment',
        use_container_width=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    if run:
        if dataset_choice == 'Both datasets':
            selected = (
                frame[
                    frame['dataset'].isin(
                        [
                            'BIRD Mini-Dev',
                            'Spider',
                        ]
                    )
                ]
                .groupby(
                    'dataset',
                    group_keys=False,
                )
                .head(
                    int(count)
                )
            )
        else:
            selected = (
                frame[
                    frame['dataset']
                    == dataset_choice
                ]
                .head(
                    int(count)
                )
            )

        if selected.empty:
            st.error(
                'No benchmark rows were found for the selected dataset.'
            )
        else:
            progress = st.progress(0)
            note = st.empty()

            def update(
                current,
                total,
                message,
            ):
                progress.progress(
                    min(
                        current / total,
                        1.0,
                    )
                )
                note.caption(
                    message
                )

            with st.spinner(
                'Running real PostgreSQL Text-to-SQL comparison...'
            ):
                try:
                    output = run_experiment(
                        selected,
                        len(selected),
                        provider,
                        update,
                    )

                    st.session_state[
                        'experiment_summary'
                    ] = output[
                        'summary'
                    ]

                    st.session_state[
                        'experiment_overall'
                    ] = output.get(
                        'overall'
                    )

                    st.success(
                        'Experiment finished successfully. Results were saved.'
                    )

                    st.subheader(
                        'Dataset and strategy summary'
                    )

                    st.dataframe(
                        output[
                            'summary'
                        ].round(4),
                        use_container_width=True,
                        hide_index=True,
                    )

                    overall = output.get(
                        'overall'
                    )

                    if (
                        overall is not None
                        and not overall.empty
                    ):
                        st.subheader(
                            'Overall strategy comparison'
                        )

                        st.dataframe(
                            overall.round(4),
                            use_container_width=True,
                            hide_index=True,
                        )

                    best = output.get(
                        'best_strategy',
                        {},
                    )

                    if best:
                        accuracy = float(
                            best.get(
                                'execution_accuracy_percent',
                                0,
                            )
                        )

                        st.success(
                            f"Best current dataset/strategy combination: "
                            f"{best.get('dataset', '')} · "
                            f"{best.get('strategy', '')} · "
                            f"Execution accuracy {accuracy:.2f}%"
                        )

                except Exception as exc:
                    st.error(
                        f'Experiment failed: {exc}'
                    )


elif page == 'Comparison':
    st.header('Experiment comparison')
    st.caption(
        'Analyse execution accuracy and efficiency across BIRD, Spider, '
        'Full Schema and relevant-schema retrieval strategies.'
    )

    results = get_results()

    if results.empty:
        st.info(
            'Run an experiment first. The Comparison page uses the saved '
            'experiment_results.csv file.'
        )

    else:
        # ----------------------------------------------------
        # Dataset / strategy summary
        # ----------------------------------------------------

        summary_path = (
            config.OUTPUT_DIR
            / 'dataset_strategy_summary.csv'
        )

        summary = (
            pd.read_csv(
                summary_path
            )
            if summary_path.exists()
            else pd.DataFrame()
        )

        if not summary.empty:
            st.subheader(
                'Dataset and strategy performance'
            )

            preferred_columns = [
                'dataset',
                'strategy',
                'questions',
                'execution_accuracy_percent',
                'sql_execution_percent',
                'exact_match_percent',
                'prompt_tokens',
                'generation_latency',
                'execution_time',
                'schema_tables',
                'repair_rate',
            ]

            available_columns = [
                column
                for column in preferred_columns
                if column in summary.columns
            ]

            st.dataframe(
                summary[
                    available_columns
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Key metrics
        # ----------------------------------------------------

        st.subheader(
            'Key research metrics'
        )

        metric_source = (
            summary
            if not summary.empty
            else results
        )

        accuracy_column = (
            'execution_accuracy'
            if 'execution_accuracy'
            in metric_source.columns
            else 'exact_match'
            if 'exact_match'
            in metric_source.columns
            else None
        )

        if accuracy_column:
            if not summary.empty:
                best_row = summary.sort_values(
                    accuracy_column,
                    ascending=False,
                ).iloc[0]

                best_accuracy = float(
                    best_row[
                        accuracy_column
                    ]
                ) * 100
            else:
                best_row = None
                best_accuracy = float(
                    metric_source[
                        accuracy_column
                    ].mean()
                ) * 100
        else:
            best_row = None
            best_accuracy = float('nan')

        full_rows = results[
            results[
                'strategy'
            ] == 'Full schema'
        ]

        top5_rows = results[
            results[
                'strategy'
            ] == 'Top-5'
        ]

        if (
            'execution_accuracy'
            in results.columns
        ):
            full_accuracy = (
                full_rows[
                    'execution_accuracy'
                ].mean()
                * 100
                if not full_rows.empty
                else float('nan')
            )

            top5_accuracy = (
                top5_rows[
                    'execution_accuracy'
                ].mean()
                * 100
                if not top5_rows.empty
                else float('nan')
            )
        else:
            full_accuracy = float('nan')
            top5_accuracy = float('nan')

        if (
            'prompt_tokens'
            in results.columns
            and not full_rows.empty
            and not top5_rows.empty
        ):
            full_tokens = (
                full_rows[
                    'prompt_tokens'
                ].mean()
            )

            top5_tokens = (
                top5_rows[
                    'prompt_tokens'
                ].mean()
            )

            token_reduction = (
                (
                    full_tokens
                    - top5_tokens
                )
                / full_tokens
                * 100
                if full_tokens
                else float('nan')
            )
        else:
            token_reduction = float('nan')

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            'Full Schema accuracy',
            (
                f'{full_accuracy:.2f}%'
                if pd.notna(
                    full_accuracy
                )
                else 'N/A'
            ),
        )

        m2.metric(
            'Top-5 accuracy',
            (
                f'{top5_accuracy:.2f}%'
                if pd.notna(
                    top5_accuracy
                )
                else 'N/A'
            ),
        )

        m3.metric(
            'Top-5 accuracy change',
            (
                f'{top5_accuracy - full_accuracy:+.2f} pp'
                if (
                    pd.notna(
                        top5_accuracy
                    )
                    and pd.notna(
                        full_accuracy
                    )
                )
                else 'N/A'
            ),
        )

        m4.metric(
            'Top-5 token reduction',
            (
                f'{token_reduction:.2f}%'
                if pd.notna(
                    token_reduction
                )
                else 'N/A'
            ),
        )

        # ----------------------------------------------------
        # Strategy accuracy table
        # ----------------------------------------------------

        if (
            'execution_accuracy'
            in results.columns
        ):
            st.subheader(
                'Execution accuracy by strategy'
            )

            strategy_accuracy = (
                results
                .groupby(
                    'strategy',
                    as_index=False,
                )
                .agg(
                    execution_accuracy=(
                        'execution_accuracy',
                        'mean',
                    ),
                    sql_execution_rate=(
                        'success',
                        'mean',
                    ),
                    prompt_tokens=(
                        'prompt_tokens',
                        'mean',
                    ),
                    generation_latency=(
                        'generation_latency',
                        'mean',
                    ),
                )
            )

            strategy_accuracy[
                'execution_accuracy_percent'
            ] = (
                strategy_accuracy[
                    'execution_accuracy'
                ]
                * 100
            )

            strategy_accuracy[
                'sql_execution_percent'
            ] = (
                strategy_accuracy[
                    'sql_execution_rate'
                ]
                * 100
            )

            st.dataframe(
                strategy_accuracy[
                    [
                        'strategy',
                        'execution_accuracy_percent',
                        'sql_execution_percent',
                        'prompt_tokens',
                        'generation_latency',
                    ]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Full vs Top-K paired significance tests
        # ----------------------------------------------------

        st.subheader(
            'Full Schema vs relevant-schema significance tests'
        )

        try:
            strategy_tests = (
                all_strategy_tests(
                    results
                )
            )

            if strategy_tests.empty:
                st.info(
                    'Not enough paired observations are available yet.'
                )

            else:
                display_test_columns = [
                    column
                    for column in [
                        'baseline',
                        'comparison',
                        'paired_questions',
                        'baseline_accuracy_percent',
                        'comparison_accuracy_percent',
                        'absolute_improvement_percent',
                        'improved_questions',
                        'worsened_questions',
                        'unchanged_questions',
                        'statistic',
                        'p_value',
                        'significant',
                    ]
                    if column
                    in strategy_tests.columns
                ]

                st.dataframe(
                    strategy_tests[
                        display_test_columns
                    ].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

                top5_test = (
                    strategy_tests[
                        strategy_tests[
                            'comparison'
                        ]
                        == 'Top-5'
                    ]
                )

                if (
                    not top5_test.empty
                    and 'p_value'
                    in top5_test.columns
                ):
                    row = (
                        top5_test.iloc[0]
                    )

                    a, b, c = st.columns(3)

                    a.metric(
                        'Full vs Top-5 p-value',
                        f"{float(row['p_value']):.4f}",
                    )

                    b.metric(
                        'Statistically significant',
                        (
                            'Yes'
                            if bool(
                                row[
                                    'significant'
                                ]
                            )
                            else 'No'
                        ),
                    )

                    c.metric(
                        'Accuracy difference',
                        (
                            f"{float(row['absolute_improvement_percent']):+.2f} pp"
                        ),
                    )

        except Exception as exc:
            st.warning(
                f'Strategy significance analysis could not be calculated: {exc}'
            )

        # ----------------------------------------------------
        # Efficiency tests
        # ----------------------------------------------------

        st.subheader(
            'Full Schema vs Top-5 efficiency'
        )

        efficiency_rows = []

        for metric, label in [
            (
                'prompt_tokens',
                'Prompt tokens',
            ),
            (
                'generation_latency',
                'Generation latency',
            ),
            (
                'schema_tables',
                'Schema tables supplied',
            ),
        ]:
            if metric not in results.columns:
                continue

            try:
                efficiency = (
                    strategy_efficiency_test(
                        results=results,
                        metric=metric,
                        baseline='Full schema',
                        comparison='Top-5',
                    )
                )

                if (
                    efficiency.get(
                        'status'
                    )
                    == 'ok'
                ):
                    efficiency_rows.append(
                        {
                            'metric': label,
                            'full_schema_mean': (
                                efficiency[
                                    'baseline_mean'
                                ]
                            ),
                            'top_5_mean': (
                                efficiency[
                                    'comparison_mean'
                                ]
                            ),
                            'absolute_change': (
                                efficiency[
                                    'absolute_change'
                                ]
                            ),
                            'relative_change_percent': (
                                efficiency.get(
                                    'relative_change_percent'
                                )
                            ),
                            'p_value': (
                                efficiency[
                                    'p_value'
                                ]
                            ),
                            'significant': (
                                efficiency[
                                    'significant'
                                ]
                            ),
                        }
                    )

            except Exception:
                continue

        if efficiency_rows:
            st.dataframe(
                pd.DataFrame(
                    efficiency_rows
                ).round(4),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                'Efficiency comparisons are not available yet.'
            )

        # ----------------------------------------------------
        # BIRD vs Spider comparison
        # ----------------------------------------------------

        st.subheader(
            'BIRD vs Spider'
        )

        try:
            dataset_test = (
                dataset_comparison_test(
                    results
                )
            )

            if (
                dataset_test.get(
                    'status'
                )
                == 'ok'
            ):
                a, b, c = st.columns(3)

                a.metric(
                    'Test',
                    dataset_test[
                        'test'
                    ],
                )

                b.metric(
                    'p-value',
                    f"{dataset_test['p_value']:.4f}",
                )

                c.metric(
                    'Significant',
                    (
                        'Yes'
                        if dataset_test.get(
                            'significant',
                            dataset_test[
                                'p_value'
                            ]
                            < 0.05,
                        )
                        else 'No'
                    ),
                )

                rates = (
                    dataset_test.get(
                        'dataset_accuracy_percent',
                        {},
                    )
                )

                if rates:
                    rate_frame = pd.DataFrame(
                        [
                            {
                                'dataset': key,
                                'execution_accuracy_percent': value,
                            }
                            for key, value
                            in rates.items()
                        ]
                    )

                    st.dataframe(
                        rate_frame,
                        use_container_width=True,
                        hide_index=True,
                    )

                interpretation = (
                    dataset_test.get(
                        'interpretation'
                    )
                )

                if interpretation:
                    st.info(
                        interpretation
                    )

            else:
                st.info(
                    dataset_test.get(
                        'message',
                        'Two datasets with both outcome classes are required.',
                    )
                )

        except Exception as exc:
            st.warning(
                f'Dataset comparison could not be calculated: {exc}'
            )

        # ----------------------------------------------------
        # Complete analysis expander
        # ----------------------------------------------------

        with st.expander(
            'Complete statistical analysis'
        ):
            try:
                analysis = (
                    complete_statistical_analysis(
                        results
                    )
                )

                st.write(
                    'Dataset comparison'
                )
                st.json(
                    analysis.get(
                        'dataset_comparison',
                        {},
                    )
                )

                strategy_frame = (
                    analysis.get(
                        'strategy_comparisons'
                    )
                )

                if (
                    isinstance(
                        strategy_frame,
                        pd.DataFrame,
                    )
                    and not strategy_frame.empty
                ):
                    st.write(
                        'Strategy comparisons'
                    )
                    st.dataframe(
                        strategy_frame.round(4),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.write(
                    'Efficiency comparisons'
                )
                st.json(
                    analysis.get(
                        'efficiency_comparisons',
                        {},
                    )
                )

            except Exception as exc:
                st.warning(
                    str(exc)
                )

        # ----------------------------------------------------
        # Download results
        # ----------------------------------------------------

        csv = (
            results
            .to_csv(
                index=False
            )
            .encode(
                'utf-8'
            )
        )

        st.download_button(
            'Download experiment results',
            csv,
            'experiment_results.csv',
            'text/csv',
        )


elif page == 'ML prediction':
    st.header('ML prediction')
    st.caption('Three machine-learning classifiers model benchmark SQL outcome patterns using grouped train/test evaluation to reduce question leakage.')
    results = get_results()
    if results.empty:
        st.info('Run enough experiments first. At least 30 completed rows containing both outcome classes are required.')
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write('Models: Logistic Regression, Random Forest and Gradient Boosting.')
        st.caption(
            'The grouped split keeps the same natural-language question out of both '
            'training and testing sets. Cross-validation and the overfitting gap are '
            'reported so a score of 1.0 can be interpreted correctly rather than assumed '
            'to mean perfect generalisation.'
        )
        if st.button('Train and compare models', use_container_width=True):
            with st.spinner('Training models...'):
                try:
                    output = train_models(results)
                    st.session_state['ml_metrics'] = output['metrics']
                    st.session_state['ml_predictions'] = output['predictions']
                    st.session_state['ml_evaluation_info'] = output.get(
                        'evaluation_info'
                    )
                except Exception as exc:
                    st.error(str(exc))
        st.markdown('</div>', unsafe_allow_html=True)
        metrics = st.session_state.get('ml_metrics')
        predictions = st.session_state.get('ml_predictions')
        evaluation_info = st.session_state.get('ml_evaluation_info')

        if evaluation_info:
            with st.expander('ML evaluation setup'):
                st.json(evaluation_info)

        if metrics is not None:
            st.subheader('Model performance')
            st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
            best = metrics.iloc[0]
            st.success(f"Best model: {best['model']} · F1 score {best['f1']:.3f}")
        if predictions is not None:
            with st.expander('Prediction details'):
                st.dataframe(predictions.head(300), use_container_width=True, hide_index=True)
