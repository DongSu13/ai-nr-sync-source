"""Read-only scientific reporting; never trains, selects, or changes thresholds."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from nr_v3_campaign import DATA


def pct(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def primary_statement(primary):
    delta=primary['delta_recovery'];lo,hi=primary['ci95']
    if delta is None or lo is None:return '主终点分母不足，不能判断增益。'
    verdict='支持主终点正增益' if lo>0 else '不支持主终点正增益'
    return f'{verdict}：AI 相对强经典方法用户恢复率差 {delta*100:+.2f} 个百分点，95% 场景配对 CI [{lo*100:+.2f}, {hi*100:+.2f}] 个百分点。'


def display_methods(lock):
    names=['correlation',lock['selected_classical']]
    names += [f'full__{n}__s22' for n in lock['family_winners'].values()]
    return list(dict.fromkeys(names))


def short(name):
    return name.replace('full__','').replace('__s22','')


def budget_table(iid,names):
    lines=['## 解码预算与候选池上限','',
           '候选池上限指离线穷举全部 12 候选可恢复的用户比例，不是预算内实际恢复率。相对上限覆盖率以池中可恢复用户为分母。','',
           '| 方法 | 预算 B | 恢复 / 发射用户 | 实际恢复率 | 候选池上限 | 相对上限覆盖率 | 总解码尝试 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for budget,b in iid['methods'][name]['budgets'].items():
            lines.append(f"| {short(name)} | {budget} | {b['recovered_users']} / {b['transmitted_users']} | {pct(b['recovery_rate'])} | {pct(b['available_coverage'])} | {pct(b['recovery_of_available'])} | {b['decode_attempts']} |")
    return '\n'.join(lines)


def detail_tables(results,lock):
    """Render recorded exploratory comparisons and measured, offline timing."""
    def points(value):
        return 'N/A' if value is None else f'{100*value:+.2f}'
    def interval(comparison):
        lo,hi=comparison.get('ci95',[None,None])
        return f'[{points(lo)}, {points(hi)}]'
    iid=results['test']
    lines=['## 逐模型与逐场景差异（探索性）','',
           '差异均相对冻结的强经典对照，单位为百分点；95% CI 为场景配对 bootstrap 的逐项区间，未做多重比较校正。三个训练种子分别列出，不视为独立无线样本。','',
           '| 方法 / 训练种子 | B=2 用户恢复率 | 差异 | 95% CI |',
           '|---|---:|---:|---:|']
    for name,method in iid['methods'].items():
        comparison=method.get('exploratory_vs_classical',{})
        lines.append(f"| {name} | {pct(method['budgets']['2']['recovery_rate'])} | {points(comparison.get('delta_recovery'))} | {interval(comparison)} |")
    lines += ['', '| 扫描单元 | AI 恢复率 | 强经典恢复率 | 差异 | 95% CI |',
              '|---|---:|---:|---:|---:|']
    for cell,values in results['sweep']['cells'].items():
        ai=values[lock['selected_ai']];base=values[lock['selected_classical']]
        comparison=ai.get('exploratory_vs_classical',{})
        lines.append(f"| {cell} | {pct(ai['budget2']['recovery_rate'])} | {pct(base['budget2']['recovery_rate'])} | {points(comparison.get('delta_recovery'))} | {interval(comparison)} |")
    lines += ['', '## 实测离线耗时','',
              '批量评分时间不包含前端、特征测量及 PBCH 解码，也不能推算为在线端到端加速；单场景时间为记录值的中位数。','',
              '| 测试集 | 候选数 | 方法 | 批量评分秒数 |','|---|---:|---|---:|']
    for split,result in results.items():
        for name,seconds in result.get('score_seconds',{}).items():
            lines.append(f"| {split} | {result.get('candidates','N/A')} | {name} | {seconds:.3f} |")
    lines += ['', '| 测试集 | 测量环节 | 单场景中位秒数 |', '|---|---|---:|']
    for split,result in results.items():
        for name,seconds in result.get('median_scene_timing',{}).items():
            lines.append(f'| {split} | {name} | {seconds:.3f} |')
    return '\n'.join(lines)


def render_report(results,lock,audit,training):
    if audit.get('status')!='PASS':raise ValueError('audit must pass before report')
    iid=results['test'];methods=display_methods(lock)
    lines=['# 多场景、多模型 NR AI 纯仿真对比（v3）','',
        '## 主结论','',primary_statement(iid['primary']),'',
        f"主测试：{iid['scenes']} 个新独立场景，{iid['candidates']} 个候选；每场景最多 2 次 PBCH 解码，分母为全部发射用户。",'',
        f"验证集选定 AI：`{lock['selected_ai']}`；强经典对照：`{lock['selected_classical']}`。",'',
        '| 方法 | 恢复 / 发射用户 | 用户恢复率 | 弱用户恢复率 | 全用户成功场景率 | 1%验证目标下候选召回 | 实测候选误接收 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in methods:
        m=iid['methods'][name];b=m['budgets']['2'];c=m['candidate']['0.01']
        lines.append(f"| {short(name)} | {b['recovered_users']} / {b['transmitted_users']} | {pct(b['recovery_rate'])} | {pct(b['weak_recovery_rate'])} | {pct(b['full_recovery_rate'])} | {pct(c['tpr'])} | {pct(c['fpr'])} |")
    lines += ['', '## 困难场景与特征消融', '',
        '以下比较是预定义辅助/探索性分析，不以未校正的多个分组检验替代主终点结论。', '',
        '| 困难组合 | AI 用户恢复率 | 强经典用户恢复率 |', '|---|---:|---:|']
    for cell,values in results['stress']['cells'].items():
        lines.append(f"| {cell} | {pct(values[lock['selected_ai']]['budget2']['recovery_rate'])} | {pct(values[lock['selected_classical']]['budget2']['recovery_rate'])} |")
    lines += ['', '| 消融模型（独立重新训练、验证选参） | B=2 用户恢复率 |', '|---|---:|']
    for name,m in iid['methods'].items():
        if name.startswith('no_') and name.endswith('__s22'):
            lines.append(f"| {short(name)} | {pct(m['budgets']['2']['recovery_rate'])} |")
    lines += ['',budget_table(iid,methods),'',detail_tables(results,lock),'']
    noise=results['noise']['noise'];warnings_count=sum(len(r['warnings']) for r in training['rows']+training['ablations'])
    lines += ['', '## 审计与解释边界','',
        f"- 数据审计通过：{audit['unique_scenes']} 个正式场景；训练/验证/开发/最终测试种子隔离。",
        '- 模型与门限在最终测试场景生成前冻结；三个训练种子全部保留，主实例预先指定为种子 22，不挑最好种子。',
        f"- 共记录 {warnings_count} 条模型训练警告；详见 training_summary.json，未删除不收敛或表现较差的模型。",
        '- 主置信区间对无线场景做配对 bootstrap；训练随机性另报，不能视为新增独立无线样本。',
        f"- 纯噪声 {noise['scenes']} 场景：离线全部候选中，有 CRC 输出的场景 {noise['scenes_with_any_crc_output']} 个，场景错误输出概率单侧 95% 上界 {pct(noise['upper95'])}。这个上界不表示已测得极低空口假警率。",
        '- 候选误接收不是最终 CRC 后误检；候选池上限与用户恢复率不同。固定 12 候选会限制拥挤和同 PSS 场景。',
        '- 这是已知 numerology/首 SSB 模式的基带仿真，非完整频率搜索、实际网络接入或硬件实测。',
        '- v3 固定前端只比较第一轮评分；不能把结果称为 AI 闭环 SIC 增益。旧 v2 的筛选增益结论单独保留。',
        '- 特征测量和离线全部候选解码时间、模型批量评分时间分别报告；没有声称在线实测加速。','',
        '## 全部结果','',
        '- metrics.csv：每种方法、预算和扫描单元的计数与比率。',
        '- 各 split_evaluation.json：三种门限、所有选定模型种子、消融、时间统计及主差异。',
        '- training_summary.json / models：全部参数配置和拟合警告，包括未选中的模型。',
        '- freeze.json：冻结参数、科学代码/模型散列和数据散列。',
        '- integrity_audit.json / *_predictions.npz：独立混淆计数和用户恢复计数复核依据。','']
    for plot in ('recovery_snr.png','roc_pr.png','ablation.png'):
        lines += [f'![{plot}]({(DATA/plot).as_posix()})','']
    return '\n'.join(lines)


def export_csv(results,path):
    rows=[]
    for split,result in results.items():
        for name,m in result['methods'].items():
            for budget,b in m['budgets'].items():
                rows.append(dict(split=split,cell='ALL',method=name,budget=int(budget),**b,
                                 tpr_1pct=m['candidate']['0.01']['tpr'],fpr_1pct=m['candidate']['0.01']['fpr']))
        for cell,values in result['cells'].items():
            for name,m in values.items():
                rows.append(dict(split=split,cell=cell,method=name,budget=2,**m['budget2'],
                                 tpr_1pct=m['candidate']['tpr'],fpr_1pct=m['candidate']['fpr']))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plots(results,lock,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    names=display_methods(lock);sweep=results['sweep']['cells'];snrs=range(-12,13,3)
    fig,ax=plt.subplots(figsize=(8,5))
    for name in names:
        ax.plot(list(snrs),[100*sweep[f'snr_db_{s}'][name]['budget2']['recovery_rate'] for s in snrs],marker='o',label=short(name))
    ax.set(xlabel='Strong configured user SNR (dB)',ylabel='Recovered / transmitted users (%)',title='First pass, two PBCH decode attempts per scene')
    ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'recovery_snr.png',dpi=160);plt.close(fig)
    data=np.load(root/'test_predictions.npz');fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name in names:
        fpr,tpr,_=roc_curve(data['y'],data[name]);precision,recall,_=precision_recall_curve(data['y'],data[name])
        axes[0].plot(fpr,tpr,label=short(name));axes[1].plot(recall,precision,label=short(name))
    axes[0].set(xlabel='Candidate false acceptance',ylabel='Decodable candidate recall',xlim=(0,.1),ylim=(0,1))
    axes[1].set(xlabel='Decodable candidate recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'roc_pr.png',dpi=160);plt.close(fig)
    methods=results['test']['methods'];selected=[lock['selected_ai']]+[n for n in methods if n.startswith('no_') and n.endswith('__s22')]
    fig,ax=plt.subplots(figsize=(9,4));labels=['Full features']+[n.split('__')[0] for n in selected[1:]]
    ax.bar(labels,[100*methods[n]['budgets']['2']['recovery_rate'] for n in selected]);ax.set(ylabel='Recovered / transmitted users (%)',title='Feature-group ablations (validation-selected hyperparameters)')
    fig.tight_layout();fig.savefig(root/'ablation.png',dpi=160);plt.close(fig)


def main():
    root=DATA;results={s:json.loads((root/f'{s}_evaluation.json').read_text()) for s in ('test','sweep','stress','noise')}
    lock=json.loads((root/'freeze.json').read_text());audit=json.loads((root/'integrity_audit.json').read_text())
    training=json.loads((root/'training_summary.json').read_text())
    report=render_report(results,lock,audit,training);export_csv(results,root/'metrics.csv');plots(results,lock,root)
    (root/'expanded_report.md').write_text(report,encoding='utf-8');print(root/'expanded_report.md')


if __name__=='__main__':main()
