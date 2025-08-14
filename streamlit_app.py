import streamlit as st
import subprocess
import os
from datetime import datetime

# Page settings
st.set_page_config(
    page_title="Adversarial Attack Toolbox",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("Adversarial Attack Toolbox")
st.markdown("---")

# User Guide (moved to the top)
st.markdown("### User Guide")

with st.expander("Click to view detailed instructions"):
    st.markdown("""
    #### Description of analytical methods

    **SPADE (robustness analysis)**
    - Evaluating model resistance to adversarial attacks
    - Score range: 0-1, closer to 1 means more robust
    - Based on documented test results: bert-base-uncased (1.0056), distilbert-base-uncased (1.0072)

    **SHAPr (Privacy risk analysis)**
    - Evaluate whether the model is prone to leaking information about the training data
    - Score range: 0-1, with higher scores indicating greater privacy risk
    - Document-based test results: Multiple models show a risk score of 1.0

    **LIME (modelling)**
    - Generating locally interpretable model prediction descriptions
    - Helps to understand the modelling decision-making process
    - Output of interpreted files in HTML format

    #### Supported datasets
    - **IMDB**: Movie Review Sentiment Analysis Dataset (2 classes: positive/negative)

    #### Supported Models
    - **bert-base-uncased**: Google BERT basic model
    - **roberta-base**: Facebook RoBERTa basic model  
    - **distilbert-base-uncased**: DistilBERT lightweight model
    - **mymodel**: Customised Local Models

    #### Important notes
    - The first run will download the model and may take a long time
    - Ensure that your internet connection is working properly to access HuggingFace!
    - The results are automatically saved to the `results/` directory.
    """)

st.markdown("---")

# Step 1: Select the type of task
st.subheader("Step 1: Select the type of task")
task_type = st.radio(
    "Select the type of data to be processed:",
    ["NLP (Natural Language Processing)", "Image (Image Processing)"],
    horizontal=True
)

# 如果选择了Image，显示图像处理配置
if task_type == "Image (Image Processing)":
    # Step 2: Image Configuration
    st.markdown("---")
    st.subheader("Step 2: Image Configuration")

    # 创建两列布局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Parameter Configuration")

        # Dataset selection for Image
        st.markdown("#### Dataset selection")
        dataset_source = st.radio(
            "Source of datasets:",
            ["HuggingFace Dataset", "Customised dataset"],
            help="Image tasks support both HuggingFace and custom datasets",
            key="image_dataset_source"
        )

        if dataset_source == "Customised dataset":
            st.markdown("##### Custom Image Dataset Upload")
            st.info("📁 Upload your custom image dataset files")

            # Image dataset upload
            st.markdown("**Required files for Image dataset:**")
            st.markdown("• **labels.csv**: CSV file with image filenames and labels")
            st.markdown("• **Image files**: Upload image files (PNG, JPG, JPEG)")

            # CSV file uploader
            labels_file = st.file_uploader(
                "Upload labels CSV file (labels.csv)",
                type=['csv'],
                help="CSV file with image filename and label columns",
                key="image_labels_file"
            )

            # Image files uploader
            image_files = st.file_uploader(
                "Upload image files",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload all image files referenced in the labels.csv",
                key="image_files"
            )

            if labels_file is not None:
                # Create mydata/image directory
                image_data_dir = "mydata/image"
                images_dir = f"{image_data_dir}/images"
                os.makedirs(images_dir, exist_ok=True)

                # Save labels file
                with open(f"{image_data_dir}/labels.csv", "wb") as f:
                    f.write(labels_file.getbuffer())

                st.success("✅ Labels file uploaded successfully!")

                # Preview the uploaded file
                import pandas as pd

                try:
                    df = pd.read_csv(labels_file)
                    st.write("**Labels data preview:**")
                    st.dataframe(df.head())
                    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

                    # Check if it has at least 2 columns
                    if df.shape[1] >= 2:
                        st.success("✅ CSV file format appears correct!")
                    else:
                        st.error("❌ CSV should have at least 2 columns (filename, label).")
                        st.stop()

                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    st.stop()

            if image_files is not None and len(image_files) > 0:
                # Save image files
                for image_file in image_files:
                    with open(f"{images_dir}/{image_file.name}", "wb") as f:
                        f.write(image_file.getbuffer())

                st.success(f"✅ {len(image_files)} image files uploaded successfully!")

            if labels_file is not None:
                hf_dataset = "custom_image"  # 标识使用自定义图像数据集
            else:
                st.warning("Please upload the labels CSV file to proceed.")
                st.stop()
        else:
            # HuggingFace dataset configuration for images
            hf_dataset = st.selectbox(
                "Choose HuggingFace Dataset:",
                ["uoft-cs/cifar10"],
                help="Currently supports CIFAR10 dataset for image tasks",
                key="image_hf_dataset"
            )

        # 模型选择 - 图像任务只支持本地模型
        st.markdown("#### Model Selection")
        st.info("📝 Image tasks currently only support custom local models")
        model_choice = st.radio(
            "Model Selection:",
            ["mymodel"],
            help="Image tasks require a custom local model defined in model.py",
            key="image_model_choice"
        )

        # 方法选择 - 图像有更多方法
        st.markdown("#### Methods of analysis")
        st.write("Selection of analytical methods (can select multiple):")

        # 使用复选框的方式进行多选
        col_clever, col_spade, col_shapr = st.columns(3)
        col_poison, col_lime, col_geex = st.columns(3)

        with col_clever:
            clever_selected = st.checkbox("CLEVER", value=False, help="Robustness evaluation", key="image_clever")

        with col_spade:
            spade_selected = st.checkbox("SPADE", value=True, help="Robustness analysis", key="image_spade")

        with col_shapr:
            shapr_selected = st.checkbox("SHAPr", value=False, help="Privacy risk analysis", key="image_shapr")

        with col_poison:
            poison_selected = st.checkbox("POISON", value=False, help="Data poisoning attack", key="image_poison")

        with col_lime:
            lime_selected = st.checkbox("LIME", value=False, help="Model explanation", key="image_lime")

        with col_geex:
            geex_selected = st.checkbox("GEEX", value=False, help="Advanced explanation", key="image_geex")

        # 根据选择构建方法列表
        analysis_methods = []
        if clever_selected:
            analysis_methods.append("CLEVER")
        if spade_selected:
            analysis_methods.append("SPADE")
        if shapr_selected:
            analysis_methods.append("SHAPr")
        if poison_selected:
            analysis_methods.append("POISON")
        if lime_selected:
            analysis_methods.append("LIME")
        if geex_selected:
            analysis_methods.append("GEEX")

        # 特殊参数设置
        st.markdown("#### Additional Parameters")

        # 图像通道数
        num_channels = st.selectbox(
            "Number of image channels:",
            [1, 3],
            index=0,
            help="1 for grayscale images (e.g., MNIST), 3 for color images (e.g., CIFAR10)",
            key="image_channels"
        )

        # 如果选择了POISON，需要额外参数
        if poison_selected:
            patch_size = st.number_input(
                "Patch size for poisoning:",
                min_value=1,
                max_value=32,
                value=8,
                help="Size of the trigger patch for data poisoning attack",
                key="image_patch_size"
            )

            check_attack_effect = st.checkbox(
                "Test attack effectiveness",
                value=True,
                help="Evaluate the effectiveness of the poisoning attack",
                key="image_test_attack"
            )
        else:
            patch_size = None
            check_attack_effect = False

    with col2:
        st.markdown("### Command Preview")


        # 根据选择生成命令
        def generate_image_commands():
            # 为每个选中的方法生成命令
            commands = []

            for method in analysis_methods:
                cmd_parts = ["python", "toolbox.py"]

                # 数据集参数
                if hf_dataset == "custom_image":
                    # 使用自定义图像数据集
                    cmd_parts.extend(["-d", "mydata"])
                else:
                    # 使用HuggingFace数据集
                    cmd_parts.extend(["-d", "hf_dataset"])
                    cmd_parts.extend(["--hf_dataset", hf_dataset])
                    cmd_parts.extend(["--hf_text_field", "img"])
                    cmd_parts.extend(["--hf_label_field", "label"])

                # 模型参数
                cmd_parts.extend(["-m", "mymodel"])

                # 任务参数
                if method == "CLEVER":
                    cmd_parts.extend(["-t", "robustness_clever"])
                elif method == "SPADE":
                    cmd_parts.extend(["-t", "robustness_spade"])
                elif method == "SHAPr":
                    cmd_parts.extend(["-t", "privacy"])
                elif method == "POISON":
                    cmd_parts.extend(["-t", "poison"])
                    if patch_size:
                        cmd_parts.extend(["-s", str(patch_size)])
                    if check_attack_effect:
                        cmd_parts.append("-test")
                elif method == "LIME":
                    cmd_parts.extend(["-t", "explain"])
                    cmd_parts.extend(["-ch", str(num_channels)])
                elif method == "GEEX":
                    cmd_parts.extend(["-t", "explain_geex"])
                    cmd_parts.extend(["-ch", str(num_channels)])

                # 类别数（固定为10）
                cmd_parts.extend(["-c", "10"])

                commands.append((method, cmd_parts))

            return commands


        # 生成并显示命令
        if analysis_methods:
            command_list = generate_image_commands()

            # 显示所有命令
            for i, (method, cmd_parts) in enumerate(command_list, 1):
                st.markdown(f"**Command {i} ({method}):**")
                command_str = " ".join(cmd_parts)
                st.code(command_str, language="bash")
        else:
            st.warning("Please select at least one analysis method")

        # 配置总览
        st.markdown("### Current Configuration")
        if analysis_methods:
            methods_str = ", ".join(analysis_methods)
            dataset_display = "Custom Image Dataset" if hf_dataset == "custom_image" else f"{hf_dataset} (HuggingFace)"
            config_info = f"""
            **Dataset**: {dataset_display}
            **Model**: {model_choice}
            **Methods of analysis**: {methods_str}
            **Image channels**: {num_channels}
            **Number of commands**: {len(analysis_methods)}
            """
            st.markdown(config_info)

            # 方法说明
            method_descriptions = {
                "CLEVER": "**CLEVER**: Evaluates model robustness against adversarial attacks using CLEVER scores",
                "SPADE": "**SPADE**: Assesses model robustness with scores ranging from 0-1",
                "SHAPr": "**SHAPr**: Evaluates privacy leakage risk with scores from 0-1",
                "POISON": "**POISON**: Performs data poisoning attacks to test model resilience",
                "LIME": "**LIME**: Generates local interpretable model explanations for images",
                "GEEX": "**GEEX**: Provides high-precision gradient-like explanations for images"
            }

            # 显示选中方法的说明
            selected_descriptions = []
            for method in analysis_methods:
                selected_descriptions.append(method_descriptions[method])

            st.info("\n\n".join(selected_descriptions))
        else:
            st.info("No analysis methods selected")

    # Step 3: Execute for Image
    st.markdown("---")
    st.subheader("Step 3: Execute")

    col_exec1, col_exec2, col_exec3 = st.columns([1, 1, 2])

    with col_exec1:
        if st.button("Operating analysis", type="primary", use_container_width=True, disabled=not analysis_methods,
                     key="image_execute"):
            st.session_state.start_execution_image = True

    with col_exec2:
        if st.button("Reconfiguration", use_container_width=True, key="image_reset"):
            st.experimental_rerun()

    with col_exec3:
        if analysis_methods:
            # 根据不同方法估算时间
            total_time = 0
            for method in analysis_methods:
                if method in ["POISON"]:
                    total_time += 10  # 投毒攻击需要更长时间
                elif method in ["LIME", "GEEX"]:
                    total_time += 5  # 解释方法需要中等时间
                else:
                    total_time += 3  # 其他方法

            st.markdown(f"**Estimated implementation time**: {len(analysis_methods)} method(s) ≈ ~{total_time} minutes")
        else:
            st.markdown("**Please select analysis method(s) first**")

    # 图像处理的执行结果区域
    if hasattr(st.session_state, 'start_execution_image') and st.session_state.start_execution_image:
        st.markdown("---")
        st.subheader("Image Analysis Results")

        # 获取要执行的命令列表
        command_list = generate_image_commands()

        # 显示即将执行的所有命令
        st.info(f"Executing {len(command_list)} image analysis method(s): {', '.join(analysis_methods)}")

        # 总进度指示
        total_progress = st.progress(0)
        overall_status = st.text(f"Preparing to execute {len(command_list)} image analyses...")

        # 存储所有结果
        all_results = {}

        try:
            for idx, (method, command) in enumerate(command_list):
                st.markdown(f"### Executing {method} Analysis ({idx + 1}/{len(command_list)})")

                # 当前任务进度
                current_progress = st.progress(0)
                current_status = st.text(f"Starting {method} analysis...")

                # 实时输出显示
                output_placeholder = st.empty()

                # 执行当前命令
                current_progress.progress(10)
                current_status.text(f"Initializing {method} analysis...")

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=os.getcwd()
                )

                # 实时收集输出
                output_lines = []
                current_progress.progress(25)
                current_status.text(f"Executing {method} analysis...")

                # 读取实时输出
                for line in iter(process.stdout.readline, ''):
                    if line:
                        output_lines.append(line.strip())

                        # 显示最新的输出（最后8行）
                        recent_output = '\n'.join(output_lines[-8:]) if len(output_lines) > 8 else '\n'.join(
                            output_lines)
                        with output_placeholder.container():
                            st.text_area(f"{method} Real-time output:", recent_output, height=150,
                                         key=f"image_output_{method}_{len(output_lines)}")

                        # 根据输出更新进度
                        line_lower = line.lower()
                        if any(keyword in line_lower for keyword in ["loading", "model", "successfully"]):
                            current_progress.progress(50)
                            current_status.text(f"{method}: Model loading...")
                        elif any(keyword in line_lower for keyword in ["processing", "computing", "calculating"]):
                            current_progress.progress(75)
                            current_status.text(f"{method}: Computing results...")
                        elif any(keyword in line_lower for keyword in ["score", "result", "final"]):
                            current_progress.progress(90)
                            current_status.text(f"{method}: Generating results...")

                # 等待进程完成
                process.wait()
                current_progress.progress(100)
                current_status.text(f"{method} analysis completed!")

                # 合并所有输出
                all_output = '\n'.join(output_lines)

                # 存储结果
                all_results[method] = {
                    'success': process.returncode == 0,
                    'output': all_output,
                    'command': ' '.join(command)
                }

                # 更新总进度
                total_progress.progress((idx + 1) / len(command_list))
                overall_status.text(f"Completed {idx + 1}/{len(command_list)} analyses")

                # 清空当前输出显示
                output_placeholder.empty()

                if process.returncode != 0:
                    st.error(f"{method} analysis failed with return code: {process.returncode}")
                    with st.expander(f"View {method} error log"):
                        st.text_area(f"{method} error output:", all_output, height=200)
                else:
                    st.success(f"{method} analysis completed successfully!")

            # 所有分析完成后显示统一结果
            st.markdown("---")
            st.subheader("🎯 Combined Image Analysis Results")
            overall_status.text("All image analyses completed! Displaying results...")

            # 为每个成功的方法显示结果
            successful_methods = [method for method, result in all_results.items() if result['success']]
            failed_methods = [method for method, result in all_results.items() if not result['success']]

            if successful_methods:
                st.success(f"Successfully completed: {', '.join(successful_methods)}")
            if failed_methods:
                st.error(f"Failed analyses: {', '.join(failed_methods)}")

            # 显示每个方法的结果
            for method, result in all_results.items():
                if result['success']:
                    st.markdown(f"### {method} Results")

                    # 提取分数
                    score_found = False
                    all_output = result['output']

                    # 根据不同方法提取相应的分数
                    if method == "CLEVER":
                        import re

                        patterns = [r'CLEVER.*?score[:\s]*([0-9.]+)', r'Untargeted CLEVER score[:\s]*([0-9.]+)']
                        for pattern in patterns:
                            matches = re.findall(pattern, all_output, re.IGNORECASE)
                            if matches:
                                try:
                                    score = float(matches[-1])
                                    st.metric(f"{method} Score", f"{score:.4f}")
                                    if score < 2.0:
                                        st.success("Model shows good robustness")
                                    elif score < 4.0:
                                        st.warning("Model robustness is moderate")
                                    else:
                                        st.error("Model is vulnerable to attacks")
                                    score_found = True
                                    break
                                except ValueError:
                                    continue

                    elif method == "SPADE":
                        import re

                        patterns = [r'SPADE Score[:\s]*([0-9.]+)']
                        for pattern in patterns:
                            matches = re.findall(pattern, all_output, re.IGNORECASE)
                            if matches:
                                try:
                                    score = float(matches[-1])
                                    st.metric(f"{method} Robustness Score", f"{score:.4f}")
                                    if score > 0.8:
                                        st.success("Model shows good robustness")
                                    elif score > 0.6:
                                        st.warning("Model robustness is moderate")
                                    else:
                                        st.error("Low model robustness")
                                    score_found = True
                                    break
                                except ValueError:
                                    continue

                    elif method == "SHAPr":
                        import re

                        patterns = [r'SHAPr.*?([0-9.]+)', r'Average SHAPr leakage[:\s]*([0-9.]+)']
                        for pattern in patterns:
                            matches = re.findall(pattern, all_output, re.IGNORECASE)
                            if matches:
                                try:
                                    score = float(matches[-1])
                                    st.metric(f"{method} Privacy Risk Score", f"{score:.4f}")
                                    if score > 0.8:
                                        st.error("High privacy leakage risk")
                                    elif score > 0.5:
                                        st.warning("Moderate privacy risk")
                                    else:
                                        st.success("Low privacy risk")
                                    score_found = True
                                    break
                                except ValueError:
                                    continue

                    elif method == "POISON":
                        import re

                        patterns = [r'Success Rate.*?([0-9.]+)', r'Test Success Rate.*?([0-9.]+)']
                        for pattern in patterns:
                            matches = re.findall(pattern, all_output, re.IGNORECASE)
                            if matches:
                                try:
                                    score = float(matches[-1])
                                    st.metric(f"{method} Attack Success Rate", f"{score:.4f}")
                                    if score > 0.8:
                                        st.error("Model is highly vulnerable to poisoning")
                                    elif score > 0.5:
                                        st.warning("Model shows moderate vulnerability")
                                    else:
                                        st.success("Model is resistant to poisoning")
                                    score_found = True
                                    break
                                except ValueError:
                                    continue

                    elif method in ["LIME", "GEEX"]:
                        if "explanation" in all_output.lower() or "saved" in all_output.lower():
                            st.success(f"{method} explanation generated successfully")
                            score_found = True

                        # 检查是否有生成的图片
                        if method == "LIME" and os.path.exists("explained_images"):
                            st.subheader("Generated LIME Explanations")
                            image_files = [f for f in os.listdir("explained_images") if f.endswith('.png')]
                            if image_files:
                                cols = st.columns(min(3, len(image_files)))
                                for i, img_file in enumerate(image_files[:3]):
                                    with cols[i]:
                                        st.image(f"explained_images/{img_file}", caption=f"LIME: {img_file}")

                        elif method == "GEEX" and os.path.exists("explained_images_geex"):
                            st.subheader("Generated GEEX Explanations")
                            image_files = [f for f in os.listdir("explained_images_geex") if f.endswith('.png')]
                            if image_files:
                                cols = st.columns(min(3, len(image_files)))
                                for i, img_file in enumerate(image_files[:3]):
                                    with cols[i]:
                                        st.image(f"explained_images_geex/{img_file}", caption=f"GEEX: {img_file}")

                    # 如果没有找到特定分数，显示检测到的数值
                    if not score_found:
                        import re

                        numbers = re.findall(r'\b\d+\.\d+\b', all_output)
                        if numbers:
                            st.write(f"**{method} - Detected values:**")
                            for num in numbers[-3:]:
                                st.write(f"• {num}")

                    # 添加查看详细日志的选项
                    with st.expander(f"View {method} detailed log"):
                        st.text_area(f"{method} complete output:", all_output, height=300)

            # Save and Reset buttons for all results
            st.markdown("### Result Actions")
            col_save, col_reset = st.columns(2)

            with col_save:
                if st.button("💾 Save All Results", type="primary", use_container_width=True, key="image_save_all"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # 保存所有结果到一个文件
                    combined_content = f"""=== Image Multi-Method Analysis Results ===
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {hf_dataset}
Model: {model_choice}
Analysis Methods: {', '.join(analysis_methods)}
Image Channels: {num_channels}
Number of categories: 10

"""

                    for method, result in all_results.items():
                        combined_content += f"""
=== {method} Analysis ===
Command: {result['command']}
Success: {result['success']}

Output:
{result['output']}

{'=' * 50}
"""

                    os.makedirs("results", exist_ok=True)
                    filename = f"image_multi_result_{'-'.join(analysis_methods).lower()}_{model_choice}_{timestamp}.txt"

                    with open(f"results/{filename}", "w", encoding="utf-8") as f:
                        f.write(combined_content)

                    st.success(f"All results saved to: `results/{filename}`")

            with col_reset:
                if st.button("🗑️ Reset All Results", use_container_width=True, key="image_reset_all"):
                    if 'start_execution_image' in st.session_state:
                        del st.session_state.start_execution_image
                    st.experimental_rerun()

        except Exception as e:
            st.error(f"An error occurred during execution: {str(e)}")
            import traceback

            st.text_area("Detailed error message:", traceback.format_exc(), height=200)

        finally:
            # 重置执行状态
            st.session_state.start_execution_image = False

            # 添加重新运行按钮
            if st.button("Run new analysis", key="image_run_new"):
                st.experimental_rerun()

    st.stop()  # 停止执行，避免显示NLP部分

# Step 2: NLP Configuration
st.markdown("---")
st.subheader("Step 2: NLP Configuration")

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Parameter Configuration")

    # Dataset selection
    st.markdown("#### Dataset selection")
    dataset_source = st.radio(
        "Source of datasets:",
        ["HuggingFace Dataset", "Customised dataset"],
        help="Currently supports HuggingFace datasets and custom dataset upload"
    )

    if dataset_source == "Customised dataset":
        st.markdown("##### Custom Dataset Upload")
        st.info("📁 Upload your custom NLP dataset files")

        # NLP dataset upload
        st.markdown("**Required files for NLP dataset:**")
        st.markdown("• **train.csv**: Training data with 'text' and 'label' columns")
        st.markdown("• **test.csv**: Testing data with 'text' and 'label' columns (optional)")

        # File uploaders
        train_file = st.file_uploader(
            "Upload training CSV file (train.csv)",
            type=['csv'],
            help="CSV file with 'text' and 'label' columns"
        )

        test_file = st.file_uploader(
            "Upload testing CSV file (test.csv) - Optional",
            type=['csv'],
            help="Optional: CSV file with 'text' and 'label' columns"
        )

        if train_file is not None:
            # Create mydata/nlp directory
            nlp_data_dir = "mydata/nlp"
            os.makedirs(nlp_data_dir, exist_ok=True)

            # Save training file
            with open(f"{nlp_data_dir}/train.csv", "wb") as f:
                f.write(train_file.getbuffer())

            st.success("✅ Training file uploaded successfully!")

            # Preview the uploaded file
            import pandas as pd

            try:
                df = pd.read_csv(train_file)
                st.write("**Training data preview:**")
                st.dataframe(df.head())
                st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

                # Check required columns
                if 'text' in df.columns and 'label' in df.columns:
                    st.success("✅ Required columns 'text' and 'label' found!")
                else:
                    st.error("❌ Missing required columns. Please ensure your CSV has 'text' and 'label' columns.")
                    st.stop()

            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.stop()

        if test_file is not None:
            # Save testing file
            with open(f"{nlp_data_dir}/test.csv", "wb") as f:
                f.write(test_file.getbuffer())
            st.success("✅ Testing file uploaded successfully!")

        if train_file is not None:
            hf_dataset = "custom_nlp"  # 标识使用自定义数据集
        else:
            st.warning("Please upload at least the training CSV file to proceed.")
            st.stop()
    else:
        # HuggingFace dataset configuration
        hf_dataset = st.selectbox(
            "Choose HuggingFace Dataset:",
            ["imdb"],
            help="Currently supports IMDB sentiment analysis datasets based on test documentation"
        )

    # 模型选择
    st.markdown("#### Model Selection")
    model_choice = st.radio(
        "Model Selection:",
        ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "mymodel"],
        help="Select the pre-trained model to be tested"
    )

    # 方法选择
    st.markdown("#### Methods of analysis")
    st.write("Selection of analytical methods (can select multiple):")

    # 使用复选框的方式进行多选
    col_spade, col_shapr, col_lime = st.columns(3)

    with col_spade:
        spade_selected = st.checkbox("SPADE", value=True, help="Robustness analysis")

    with col_shapr:
        shapr_selected = st.checkbox("SHAPr", value=False, help="Privacy risk analysis")

    with col_lime:
        lime_selected = st.checkbox("LIME", value=False, help="Model explanation")

    # 根据选择构建方法列表
    analysis_methods = []
    if spade_selected:
        analysis_methods.append("SPADE")
    if shapr_selected:
        analysis_methods.append("SHAPr")
    if lime_selected:
        analysis_methods.append("LIME")

with col2:
    st.markdown("### Command Preview")


    # 根据选择生成命令
    def generate_commands():
        # 为每个选中的方法生成命令
        commands = []

        for method in analysis_methods:
            cmd_parts = ["python", "toolbox.py"]

            # 数据集参数
            if hf_dataset == "custom_nlp":
                # 使用自定义NLP数据集
                cmd_parts.extend(["-d", "mydata"])
            else:
                # 使用HuggingFace数据集
                cmd_parts.extend(["-d", "hf_dataset"])
                cmd_parts.extend(["--hf_dataset", hf_dataset])
                cmd_parts.extend(["--hf_text_field", "text"])
                cmd_parts.extend(["--hf_label_field", "label"])

            # 模型参数
            if model_choice == "mymodel":
                cmd_parts.extend(["-m", "mymodel"])
            else:
                cmd_parts.extend(["-m", "AutoModelForSequenceClassification"])
                cmd_parts.extend(["--model_name", model_choice])

            # 任务参数
            if method == "SPADE":
                cmd_parts.extend(["-t", "robustness_spade_NLP"])
            elif method == "SHAPr":
                cmd_parts.extend(["-t", "privacy_NLP"])
            elif method == "LIME":
                cmd_parts.extend(["-t", "explain_nlp"])

            # 类别数（固定为10）
            cmd_parts.extend(["-c", "10"])

            commands.append((method, cmd_parts))

        return commands


    # 生成并显示命令
    if analysis_methods:
        command_list = generate_commands()

        # 显示所有命令
        for i, (method, cmd_parts) in enumerate(command_list, 1):
            st.markdown(f"**Command {i} ({method}):**")
            command_str = " ".join(cmd_parts)
            st.code(command_str, language="bash")
    else:
        st.warning("Please select at least one analysis method")

    # 配置总览
    st.markdown("### current configuration")
    if analysis_methods:
        methods_str = ", ".join(analysis_methods)
        dataset_display = "Custom NLP Dataset" if hf_dataset == "custom_nlp" else f"{hf_dataset} (HuggingFace)"
        config_info = f"""
        **Dataset**: {dataset_display}
        **Model**: {model_choice}
        **Methods of analysis**: {methods_str}
        **Number of commands**: {len(analysis_methods)}
        """
        st.markdown(config_info)

        # 方法说明
        method_descriptions = {
            "SPADE": "**SPADE**: Evaluates the robustness of the model, with scores ranging from 0-1, the closer to 1 the more robust it is.",
            "SHAPr": "**SHAPr**: Assesses the risk of a privacy breach with a score ranging from 0-1, the higher the risk, the greater the risk.",
            "LIME": "**LIME**: Generate model explanations to help understand the model decision-making process"
        }

        # 显示选中方法的说明
        selected_descriptions = []
        for method in analysis_methods:
            selected_descriptions.append(method_descriptions[method])

        st.info("\n\n".join(selected_descriptions))
    else:
        st.info("No analysis methods selected")

# Step 3: Execute
st.markdown("---")
st.subheader("Step 3: Execute")

col_exec1, col_exec2, col_exec3 = st.columns([1, 1, 2])

with col_exec1:
    if st.button("Operating analysis", type="primary", use_container_width=True, disabled=not analysis_methods):
        st.session_state.start_execution = True

with col_exec2:
    if st.button("Reconfiguration", use_container_width=True):
        st.experimental_rerun()

with col_exec3:
    if analysis_methods:
        total_time = len(analysis_methods) * 3  # 估计每个方法3分钟
        st.markdown(
            f"**Estimated implementation time**: {len(analysis_methods)} method(s) × ~3 minutes = ~{total_time} minutes")
    else:
        st.markdown("**Please select analysis method(s) first**")

# 执行结果区域
if hasattr(st.session_state, 'start_execution') and st.session_state.start_execution:
    st.markdown("---")
    st.subheader("Implementation results")

    # 获取要执行的命令列表
    command_list = generate_commands()

    # 显示即将执行的所有命令
    st.info(f"Executing {len(command_list)} analysis method(s): {', '.join(analysis_methods)}")

    # 总进度指示
    total_progress = st.progress(0)
    overall_status = st.text(f"Preparing to execute {len(command_list)} analyses...")

    # 存储所有结果
    all_results = {}

    try:
        for idx, (method, command) in enumerate(command_list):
            st.markdown(f"### Executing {method} Analysis ({idx + 1}/{len(command_list)})")

            # 当前任务进度
            current_progress = st.progress(0)
            current_status = st.text(f"Starting {method} analysis...")

            # 实时输出显示
            output_placeholder = st.empty()

            # 执行当前命令
            current_progress.progress(10)
            current_status.text(f"Initializing {method} analysis...")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.getcwd()
            )

            # 实时收集输出
            output_lines = []
            current_progress.progress(25)
            current_status.text(f"Executing {method} analysis...")

            # 读取实时输出
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line.strip())

                    # 显示最新的输出（最后8行）
                    recent_output = '\n'.join(output_lines[-8:]) if len(output_lines) > 8 else '\n'.join(output_lines)
                    with output_placeholder.container():
                        st.text_area(f"{method} Real-time output:", recent_output, height=150,
                                     key=f"output_{method}_{len(output_lines)}")

                    # 根据输出更新进度
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in ["loading", "model", "successfully"]):
                        current_progress.progress(50)
                        current_status.text(f"{method}: Model loading...")
                    elif any(keyword in line_lower for keyword in ["processing", "computing", "calculating"]):
                        current_progress.progress(75)
                        current_status.text(f"{method}: Computing results...")
                    elif any(keyword in line_lower for keyword in ["score", "result", "final"]):
                        current_progress.progress(90)
                        current_status.text(f"{method}: Generating results...")

            # 等待进程完成
            process.wait()
            current_progress.progress(100)
            current_status.text(f"{method} analysis completed!")

            # 合并所有输出
            all_output = '\n'.join(output_lines)

            # 存储结果
            all_results[method] = {
                'success': process.returncode == 0,
                'output': all_output,
                'command': ' '.join(command)
            }

            # 更新总进度
            total_progress.progress((idx + 1) / len(command_list))
            overall_status.text(f"Completed {idx + 1}/{len(command_list)} analyses")

            # 清空当前输出显示
            output_placeholder.empty()

            if process.returncode != 0:
                st.error(f"{method} analysis failed with return code: {process.returncode}")
                with st.expander(f"View {method} error log"):
                    st.text_area(f"{method} error output:", all_output, height=200)
            else:
                st.success(f"{method} analysis completed successfully!")

        # 所有分析完成后显示统一结果
        st.markdown("---")
        st.subheader("🎯 Combined Analysis Results")
        overall_status.text("All analyses completed! Displaying results...")

        # 为每个成功的方法显示结果
        successful_methods = [method for method, result in all_results.items() if result['success']]
        failed_methods = [method for method, result in all_results.items() if not result['success']]

        if successful_methods:
            st.success(f"Successfully completed: {', '.join(successful_methods)}")
        if failed_methods:
            st.error(f"Failed analyses: {', '.join(failed_methods)}")

        # 显示每个方法的结果
        for method, result in all_results.items():
            if result['success']:
                st.markdown(f"### {method} Results")

                # 提取分数
                score_found = False
                all_output = result['output']

                if method == "SPADE":
                    import re

                    patterns = [
                        r'SPADE Score[:\s]*([0-9.]+)',
                        r'spade score[:\s]*([0-9.]+)',
                        r'Score[:\s]*([0-9.]+)',
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, all_output, re.IGNORECASE)
                        if matches:
                            try:
                                score = float(matches[-1])
                                st.metric(f"{method} Robustness Score", f"{score:.4f}")

                                if score > 1.0:
                                    st.success("Model shows good robustness")
                                elif score > 0.95:
                                    st.warning("Model robustness is moderate.")
                                else:
                                    st.error("Low model robustness")

                                score_found = True
                                break
                            except ValueError:
                                continue

                elif method == "SHAPr":
                    import re

                    patterns = [
                        r'SHAPr Score[:\s]*([0-9.]+)',
                        r'shapr.*?([0-9.]+)',
                        r'Average SHAPr leakage[:\s]*([0-9.]+)',
                        r'leakage[:\s]*([0-9.]+)',
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, all_output, re.IGNORECASE)
                        if matches:
                            try:
                                score = float(matches[-1])
                                st.metric(f"{method} Privacy Risk Score", f"{score:.4f}")

                                if score > 0.8:
                                    st.error("High risk of privacy breach")
                                elif score > 0.5:
                                    st.warning("Moderate privacy risk")
                                else:
                                    st.success("Low privacy risk")

                                score_found = True
                                break
                            except ValueError:
                                continue

                elif method == "LIME":
                    if "explanation" in all_output.lower():
                        st.success("LIME explanation generated")
                        score_found = True

                    import re

                    patterns = [
                        r'LIME Score[:\s]*([0-9.]+)',
                        r'explanation.*?([0-9.]+)',
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, all_output, re.IGNORECASE)
                        if matches:
                            try:
                                score = float(matches[-1])
                                st.metric(f"{method} Explanation Strength", f"{score:.4f}")
                                break
                            except ValueError:
                                continue

                # 如果没有找到特定分数，显示检测到的数值
                if not score_found:
                    import re

                    numbers = re.findall(r'\b\d+\.\d+\b', all_output)
                    if numbers:
                        st.write(f"**{method} - Detected values:**")
                        for num in numbers[-3:]:
                            st.write(f"• {num}")

                # 添加查看详细日志的选项
                with st.expander(f"View {method} detailed log"):
                    st.text_area(f"{method} complete output:", all_output, height=300)

        # Save and Reset buttons for all results
        st.markdown("### Result Actions")
        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button("💾 Save All Results", type="primary", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 保存所有结果到一个文件
                combined_content = f"""=== NLP Multi-Method Analysis Results ===
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {hf_dataset}
Model: {model_choice}
Analysis Methods: {', '.join(analysis_methods)}
Number of categories: 10

"""

                for method, result in all_results.items():
                    combined_content += f"""
=== {method} Analysis ===
Command: {result['command']}
Success: {result['success']}

Output:
{result['output']}

{'=' * 50}
"""

                os.makedirs("results", exist_ok=True)
                filename = f"nlp_multi_result_{'-'.join(analysis_methods).lower()}_{model_choice}_{timestamp}.txt"

                with open(f"results/{filename}", "w", encoding="utf-8") as f:
                    f.write(combined_content)

                st.success(f"All results saved to: `results/{filename}`")

        with col_reset:
            if st.button("🗑️ Reset All Results", use_container_width=True):
                if 'start_execution' in st.session_state:
                    del st.session_state.start_execution
                st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred during execution: {str(e)}")
        import traceback

        st.text_area("Detailed error message:", traceback.format_exc(), height=200)

    finally:
        # 重置执行状态
        st.session_state.start_execution = False

        # 添加重新运行按钮
        if st.button("Run new analysis"):
            st.experimental_rerun()

# 侧边栏信息
st.sidebar.title("Test Statistics")
st.sidebar.markdown("### Document-based test results")

test_results = {
    "SPADE-NLP": {
        "bert-base-uncased": "1.0056",
        "distilbert-base-uncased": "1.0072",
        "roberta-base": "1.0073"
    },
    "SHAPr-NLP": {
        "bert-base-uncased": "1.0",
        "distilbert-base-uncased": "1.0",
        "roberta-base": "1.0"
    }
}

for method, results in test_results.items():
    st.sidebar.markdown(f"**{method}**")
    for model, score in results.items():
        st.sidebar.text(f"• {model}: {score}")

st.sidebar.markdown("---")
st.sidebar.info("These results are taken from the project test documentation and can be used as a reference benchmark")