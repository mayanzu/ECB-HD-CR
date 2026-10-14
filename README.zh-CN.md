# ECB-HD-CR 公开复现包

这是从修订工程中筛选出的独立仿真包，版本 1.0.0，整理日期 2026-10-14。
不含论文、标红稿、审稿回复、修订记录、参数搜索脚本或临时实验。

安装说明见英文 `README.md`。在包根目录执行：

```powershell
python verify_package.py
python reproduce.py --workers 4
```

该入口会校验文件、还原原始结果、运行 18 项实现不变量检查，从原始记录
重新导出统计表并与随包 28 个 CSV 比较，生成 SVG/PDF 图。输出在 `artifacts/`。
只检查数值、不生成图片时用 `python reproduce.py --workers 4 --no-figures`。

`results/raw_runs.zip` 保留原有正式实验 950 组和新增通信开销实验 300 组。
120 组 NSRDB 结果中的本机路径改为包内相对路径，数值未改；其余 1,130 组
原始记录的字节保持一致。辐照输入仅保留实验实际使用的前 500 条观测及
原始表头/元数据，删去了未使用的年度数据，来源与校验值见 `data/provenance.json`。

快速运行全部六种方法/五种场景的 60 轮测试：

```powershell
python experiments/run_review_revision.py --workers 4 --smoke
```

完整仿真使用两个正式实验入口，详见英文说明。建议在另一份新解压目录
运行完整仿真，以保留随包参考结果；运行器会写入同名结果文件。
包内 `.pkl` 是逐次实验记录，不含训练权重检查点。

原图 SVG/PDF 放在 `assets/figures/`，保留原来的布局，没有重绘机制图。
其他历史图片只保留现有 PDF；新增定量图可通过脚本生成 SVG 后转 PDF。

本包公开于 GitHub：https://github.com/mayanzu/ECB-HD-CR。目前没有 Zenodo DOI。原有代码 MIT 许可保留，
NSRDB 数据按其上游条款处理。CITATION.cff 使用论文实际作者信息，未加入邮箱、
本机账号或本地绝对路径。论文和回复的可用性声明应引用上述公开仓库；
持久归档标识仅在实际分配后填写。
