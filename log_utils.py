import shutil
import glob
import json
import re
import os

def move_logs_if_passed(new_directory, passed_keyword):
    current_directory = os.getcwd()
    os.makedirs(new_directory, exist_ok=True)
    for filename in os.listdir(current_directory):
        file_path = os.path.join(current_directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.log'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                if passed_keyword in content:
                    shutil.move(file_path, os.path.join(new_directory, filename))
                    print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def filter_log_file(log_file_path, exclude_keywords):
    with open(log_file_path, "r") as file:
        lines = file.readlines()
    filtered_lines = [line for line in lines if not any(keyword in line for keyword in exclude_keywords)]
    for line in filtered_lines:
        print(line, end="")

def find_funcall_file_error(error_patterns, mysys_id_pattern):
    log_files = glob.glob("*.log")
    result = []
    for log_file in log_files:
        with open(log_file, 'r') as file:
            content = file.readlines()
            if any(re.search(pattern, line) for pattern in error_patterns for line in content):
                mysys_ids = []
                for line in content:
                    match = re.search(mysys_id_pattern, line)
                    if match:
                        mysys_ids.append(match.group(1))
                if mysys_ids:
                    result.append({
                        "log_file": log_file,
                        "mysys_ids": mysys_ids
                    })
    for entry in result:
        print(f"\nLog file: {entry['log_file']}")
        for mysys_id in entry['mysys_ids']:
            print(f"mysys_id: {mysys_id}")

def parse_json(json_data, retry_on_fail=True):
    try:
        data = json.loads(json_data)
        return data
    except json.decoder.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        if retry_on_fail:
            corrected_json_data = json_data.replace('\\', '\\\\')
            try:
                data = json.loads(corrected_json_data)
                return data
            except json.decoder.JSONDecodeError as e:
                print(f"JSON decoding still failed after retry: {e}")
                return []
        else:
            return []

def move_patterns_logs(destination_path, patterns):
    current_directory = os.getcwd()
    log_files = glob.glob("*.log")
    for log_file in log_files:
        with open(log_file, 'r') as file:
            if any(re.search(pattern, line) for pattern in patterns for line in file):
                shutil.move(os.path.join(current_directory, log_file), destination_path)
                break

def split_log_file(input_file, split_pattern, output_pattern):
    with open(input_file, 'r') as file:
        log_content = file.read()
    pattern = re.compile(split_pattern)
    split_points = [match.start() for match in re.finditer(pattern, log_content)]
    split_points.append(len(log_content))
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        segment = log_content[start:end]
        match = pattern.search(segment)
        if match:
            number = match.group(1)
            output_file = output_pattern.format(number=number)
            with open(output_file, 'w') as file:
                file.write(segment)
            print(f"Segment saved as {output_file}")

# Example usage
move_logs_if_passed("/Users/emacspy/Desktop/passed_log", "======================== 1 passed")
filter_log_file("995.log", ["LLM", "emit called: "])
find_funcall_file_error(["run process erro: xxx", "o such file or directory"], r'xxxx_id: (\d+)')
move_patterns_logs("noactive_logs", [r"^xxxx.Exception", r"^run process erro"])
split_log_file("/Users/emacspy/Desktop/xxxx.log", r'Run tests/swe_(\d+).py ===============', "swe_{number}.log")

