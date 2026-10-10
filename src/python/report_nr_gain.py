"""Render only recorded holdout evidence; no model fitting or threshold tuning."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from nr_gain_evaluation import DATA

def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f} pp'
def ci(x):return f'[{100*x[0]:+.2f}, {100*x[1]:+.2f}] pp'

def main():
    result=json.loads((DATA/'test_evaluation.json').read_text());ops=json.loads((DATA/'test_operating_points.json').read_text())
    audit=json.loads((DATA/'integrity_audit.json').read_text());assert audit['status']=='PASS'
    p=ops['0.01'];a=p['selected_ai'];b=p['selected_classical'];comp=p['comparisons'];m=p['metrics']
    lines=['# 多用户 NR 同步：AI 候选筛选纯仿真结果','',
      '## 结论','',
      f"独立测试共 {result['scenes']} 个场景、{result['candidates']} 个候选，其中 {result['positive']} 个候选经 CRC 和载荷真值校验可正确恢复。",
      f"验证集 1% 候选误接收率约束下，冻结 AI 模型为 `{a}`，强经典对照为 `{b}`。",'',
      '| 方法 | TP | FP | TN | FN | 可解码候选召回率 | 候选误接收率 | AP |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['correlation',b,a]:
        r=m[name];lines.append(f"| {name} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | {pct(r['tpr'])} | {pct(r['fpr'])} | {r['average_precision']:.4f} |")
    lines+=['','召回率与误接收率之差均按同一场景配对，采用 2,000 次场景级 bootstrap；同场景候选不会当成独立实验重复计数。','']
    for name in ['correlation',b]:
        c=comp[name];lines+=[f"- AI 相对 `{name}`：召回差 {pp(c['delta_tpr'])}，95% CI {ci(c['delta_tpr_ci95'])}；误接收差 {pp(c['delta_fpr'])}，95% CI {ci(c['delta_fpr_ci95'])}。5pp召回增益判据：{'满足' if c['recall_gain_5pp'] else '未满足'}。"]
    lines+=['','## 10%工作点','',
       '以下报告验证集10%误接收约束下的冻结工作点；模型和门限应在测试数据生成前冻结，具体顺序以审计记录为准。',
       f"10%工作点相对强经典基线的5pp召回判据：{'满足' if result['gain_criterion_pass'] else '未满足'}。",'',
       '| 方法 | 10%约束下测试召回率 | 测试候选误接收率 |','|---|---:|---:|']
    for name in ['correlation',b,a]:
        r=ops['0.1']['metrics'][name];lines.append(f"| {name} | {pct(r['tpr'])} | {pct(r['fpr'])} |")
    c=ops['0.1']['comparisons'][b]
    lines+=['',f"10%工作点 AI 相对强经典基线：召回差 CI {ci(c['delta_tpr_ci95'])}，误接收差 CI {ci(c['delta_fpr_ci95'])}。召回非劣2pp且误接收显著下降判据：{'满足' if c['reduced_fp_noninferior_recall'] else '未满足'}。",'',
       '## 固定解码预算的排序用途','',
       '以下仅统计第一轮、每场景最多尝试1或2个不同PCI候选的PBCH解码。所有方法共享完全相同候选；离线先解码全部候选只为提供正确性真值。','',
       '| 方法 | 每场景预算 | 恢复用户总数 | 候选池内可恢复用户数 | 可恢复用户覆盖率 |','|---|---:|---:|---:|---:|']
    for name in ['correlation',b,a]:
        for r in result['first_pass_budget'][name]:
            lines.append(f"| {name} | {r['budget_per_scene']} | {r['confirmed_users']} | {r['available_confirmable_users']} | {pct(r['fraction_of_available'])} |")
    lines+=['','## 数据和测量边界','',
      '训练500、验证150、开发检查150、最终留出800个独立场景。包含AWGN与独立逐用户TDL-C、2/3用户、100/300ns多径、30Hz Doppler、随机PCI、±500Hz CFO、64..500样点时延、−9..0dB功率差及−6/0/6/12dB SNR。SNR按强用户活跃SSB区间功率定义。',
      '13项特征包含相关度、实际峰/旁瓣中位数比、DM-RS相干度/拟合误差、均衡后LLR统计等。所有真假候选走同一个测量函数；CRC、载荷、场景、PCI、用户真值不进入模型。',
      '这里的正例是“已生成候选中PBCH可正确恢复”，不等同于所有发射用户；误接收指筛选器接受不可正确恢复候选，不是最终CRC后的无线小区假警。AI不能突破穷举全部候选解码的上限，增益体现在筛选和受限解码预算的排序。',
      '基带链采用已知15kHz numerology和首个Case A SSB，未覆盖完整频率栅格搜索/全部SSB索引。随机24bit BCH载荷用于比特恢复验证。没有进行硬件部署。',
      '完整审计状态与源文件散列记录见 integrity_audit.json；本报告要求该文件的 status 为 PASS。',
      '', '## 全部方法与分场景结果','',
      '全部模型（含未选中的逻辑回归正则参数、两种浅层提升树）、逐场景分组及原始计数保存在 JSON/CSV 中。未依据测试结果更换选定模型或阈值。','',
      '- `test_evaluation.json`：工作点、分信道/SNR统计、解码预算。',
      '- `test_operating_points.json`：两个验证工作点、配对CI和判据。',
      '- `frozen_manifest.json` / `operating_points_lock.json`：冻结参数与源文件散列。',
      '- `integrity_audit.json`：1600场景数据审计。',
      '- `test_predictions.npz`：完整候选分数与分组，供复核。',
      '- `test/`：800个独立仿真原始文件。',
      '', f"![ROC]({(DATA/'roc_holdout.png').as_posix()})",'']
    (DATA/'AI增益_独立测试报告.md').write_text('\n'.join(lines),encoding='utf-8')
    with (DATA/'operating_point_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['validation_fpr_target','method','tp','fp','tn','fn','tpr','fpr','threshold','average_precision','auc']);writer.writeheader()
        for target,point in ops.items():
            for name,r in point['metrics'].items():writer.writerow(dict(validation_fpr_target=target,method=name,**r))
    pred=np.load(DATA/'test_predictions.npz');fig,ax=plt.subplots(figsize=(7,4.5))
    for name in ['correlation',b,a]:
        fpr,tpr,_=roc_curve(pred['y'],pred[name]);ax.plot(fpr,tpr,label=name)
    ax.set(xlim=(0,.05),ylim=(.4,1.005),xlabel='Candidate false acceptance rate',ylabel='Recall of decodable candidates',title='800 unseen NR multi-cell scenarios')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(DATA/'roc_holdout.png',dpi=180);plt.close(fig)
    print(DATA/'AI增益_独立测试报告.md')

if __name__=='__main__':main()
