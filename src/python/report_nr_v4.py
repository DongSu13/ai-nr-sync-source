"""Read-only reporting of frozen v4 scientific outputs."""
import csv
import json
from nr_v4_campaign import DATA
from nr_v4_audit import require
from nr_v4_results import primary_statement


def pct(x):return 'N/A' if x is None else f'{100*x:.3f}%'


def delta(x):return 'N/A' if x is None else f'{100*x:+.3f}'


def render_report(results,lock,audit,training):
    require(audit.get('status')=='PASS','audit must pass before report')
    main=results['test'];rows=['# NR AI 闭环 SIC 纯仿真对比（v4）','',
        '## 主结论','',primary_statement(main['primary']),'',
        '主终点：2,000 个新独立同分布场景，每场景最多6次PBCH Polar解码、4次完整候选搜索；分母为全部发射用户。',
        '主比较为预指定种子22的AI排序+质量门控SIC，对验证集选出的强经典SIC。', '',
        f"经典对照：`{lock['selected_classical']['policy']}`；AI排序模型：`{lock['selected_rank']}`；质量门控：`{lock['selected_gate']['config']}`。",'',
        '| 方法 / 解码预算 | 正确恢复 / 发射用户 | 用户恢复率 | 弱用户恢复率 | 全用户恢复场景率 | 相对经典差值（百分点） | 95% CI |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in main['methods'].items():
        c=m['exploratory_vs_classic'];lo,hi=c['ci95']
        rows.append(f"| {name} | {m['recovered_users']} / {m['transmitted_users']} | {pct(m['recovery_rate'])} | {pct(m['weak_recovery_rate'])} | {pct(m['all_recovered_rate'])} | {delta(c['delta'])} | [{delta(lo)}, {delta(hi)}] |")
    rows += ['', '除预定义主比较外，上表其他种子、预算和消融均为探索性比较；逐项区间未作多重比较校正。','',
        '## 困难场景（探索性）','',
        'stress_1：6用户/-6dB/18dB功率差；stress_2：同PSS近同步；stress_3：TDL-D/300Hz；stress_4：1000ns扩展/CFO±1500Hz。','',
        '| 场景 | 用户数分母 | AI B6恢复率 | 经典B6恢复率 | 差值（百分点） | 95% CI |','|---|---:|---:|---:|---:|---:|']
    for cell,methods in results['stress']['cells'].items():
        a=methods['ai_full_s22__B6'];b=methods['classic__B6'];c=a['exploratory_vs_classic'];lo,hi=c['ci95']
        rows.append(f"| {cell} | {a['transmitted_users']} | {pct(a['recovery_rate'])} | {pct(b['recovery_rate'])} | {delta(c['delta'])} | [{delta(lo)}, {delta(hi)}] |")
    rows += ['', '## 实际资源使用（同分布测试）','',
        '| 方法 / 预算 | 平均解码次数 | 平均搜索次数 | 平均重构次数 | 平均在线秒数 | 平均特征秒数 | 平均解码秒数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in main['methods'].items():
        rows.append(f"| {name} | {m['mean_attempts']:.2f} | {m['mean_searches']:.2f} | {m['mean_reconstructs']:.2f} | {m['mean_seconds']:.4f} | {m['mean_features_seconds']:.4f} | {m['mean_decode_seconds']:.4f} |")
    rows += ['', '## 纯噪声错误输出','',
        '| 方法（B6） | 有错误CRC输出的场景 / 场景数 | 概率单侧95%上界 |','|---|---:|---:|']
    for name,m in results['noise']['methods'].items():
        if name.endswith('__B6'):rows.append(f"| {name} | {m['false_output_scenes']} / {m['scenes']} | {pct(m['false_output_upper95'])} |")
    warning_count=sum(len(m['warnings']) for m in training['models'].values())
    rows += ['', '## 审计、训练和解释边界','',
        f"- {audit['unique_scenes']} 个正式独立场景审计PASS；训练/开发采集、验证选型和新测试种子隔离；所有测试生成时间晚于冻结。",
        f"- 保留全部 {len(training['models'])} 次模型拟合及 {warning_count} 条训练警告。全部验证配置位于各 selection_validation_summary.json，不只保存胜者。",
        '- 搜索只提取测量特征/LLR，不预先执行Polar；正式测试逐次实际解码，CRC失败禁止消除。',
        '- 消除器只使用恢复载荷和接收端信道/定时/CFO估计；各方法共用同一个重构器。',
        '- 质量门控学的是最多两步的消除/跳过分支收益，不是最优长期决策；训练状态由经典与随机探索生成，可能与AI状态分布有差异。',
        '- 训练分支中的负收益反映该短时域教师下的有害决策；不能把测试残差能量下降直接解释为消除正确。',
        '- 同上限预算不等于相同实际计算时间；时间包含在线搜索、测量、评分、解码和重构，不包含训练离线分支标签生成。',
        '- 已知numerology/首SSB、独立PCI和随机BCH载荷的基带仿真，不涉及真实网络、射频或FPGA；不能声称任意信道普遍优越。',
        '- v3固定前端结果不与v4不同场景和预算作直接增益相减；本轮保留正、零和负结果。','']
    for name in ('recovery_budget.png','weak_recovery_budget.png','stress_recovery.png'):
        rows += [f'![{name}]({(DATA/name).as_posix()})','']
    return '\n'.join(rows)


def plots(results,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('classic','ai_rank','ai_full_s22','no_sic');budgets=(2,4,6,8)
    for metric,file,title in (('recovery_rate','recovery_budget.png','All transmitted users'),
                              ('weak_recovery_rate','weak_recovery_budget.png','Configured weakest users')):
        fig,ax=plt.subplots(figsize=(8,5))
        for name in names:
            ax.plot(budgets,[100*results['test']['methods'][f'{name}__B{b}'][metric] for b in budgets],marker='o',label=name)
        ax.set(xlabel='Maximum Polar decode attempts per scene',ylabel='Correct recovery (%)',title=title)
        ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(root/file,dpi=160);plt.close(fig)
    cells=list(results['stress']['cells']);fig,ax=plt.subplots(figsize=(8,5))
    for name in names:
        ax.plot(cells,[100*results['stress']['cells'][c][f'{name}__B6']['recovery_rate'] for c in cells],marker='o',label=name)
    ax.set(ylabel='Correct recovery (%)',title='Predefined stress scenarios; exploratory, B=6')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(root/'stress_recovery.png',dpi=160);plt.close(fig)


def main():
    results={s:json.loads((DATA/f'{s}_evaluation.json').read_text()) for s in ('test','stress','noise')}
    lock=json.loads((DATA/'freeze.json').read_text());audit=json.loads((DATA/'integrity_audit.json').read_text())
    training=json.loads((DATA/'training_summary.json').read_text());report=render_report(results,lock,audit,training)
    rows=[]
    for split,result in results.items():
        for cell,methods in {'ALL':result['methods'],**result['cells']}.items():
            for name,m in methods.items():
                c=m['exploratory_vs_classic'];rows.append(dict(split=split,cell=cell,method=name,
                    **{k:v for k,v in m.items() if k!='exploratory_vs_classic'},delta=c['delta'],ci_low=c['ci95'][0],ci_high=c['ci95'][1]))
    with (DATA/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plots(results,DATA);(DATA/'closed_loop_report.md').write_text(report,encoding='utf-8')
    print(DATA/'closed_loop_report.md')


if __name__=='__main__':main()
