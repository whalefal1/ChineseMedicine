import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis.main import Main

# 健康状况判定函数
def analyze_health(health_dict):
    res = []
    if abs(health_dict.get('heart', 0)) > 0.4:
        res.append('血虚')
    if abs(health_dict.get('spleen', 0)) > 0.25:
        res.append('脾虚')
    if abs(health_dict.get('kidney', 0)) > 0.4:
        res.append('肾虚')
    if abs(health_dict.get('liver', 0)) > 0.45:
        res.append('肝郁')
    if not res:
        res.append('健康')
    return res

if __name__ == '__main__':
    Main()
    print('舌体五区健康值计算完成。')

    # 读取health_scores.txt并判定健康状况
    input_file = 'health_scores.txt'
    output_file = 'health_diagnosis.txt'
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        header = fin.readline()  # 跳过表头
        fout.write('图片名\t健康问题\n')
        for line in fin:
            items = line.strip().split('\t')
            if len(items) < 6:
                continue
            filename = items[0]
            health_dict = {
                'heart': float(items[1]),
                'kidney': float(items[2]),
                'liver': (float(items[3]) + float(items[4])) / 2,
                'spleen': float(items[5])
            }
            problems = analyze_health(health_dict)
            fout.write(f'{filename}\t{ ", ".join(problems) }\n')
    print('健康状况判定已输出到 health_diagnosis.txt') 