DEAP 数据库

EEG_feature.txt 包含了1216个脑电信号样本的160维特征，每行为一个样本，每列为一种特征。特征从左至右分别是每个脑电电极的theta（1-32列）、slow alpha（33-64列）、alpha（65-96列）、beta（1-97128列）、gamma（129-160列）波段的脑电特征。

subject_video.txt 包含了1216个脑电信号对应的32名被试和38段视频信息，其中包含两列。第一列是对应的被试编号，第二列是对应的视频编号。

EEG_feature.txt 与 subject_video.txt和valence_arousal_label.txt中每行都是一一对应的，例如subject_video.txt的第二行就是EEG_feature.txt中第二个样本（第二行）的被试和视频信息；valence_arousal_label.txt的第二行也是EEG_feature.txt中第二个样本（第二行）的愉悦度和唤醒度标签。valence_arousal_label.txt中第一列为愉悦度标签，1代表positive，2代表negative；第二列为唤醒度标签，1代表high，2代表low。

DEAP数据库并未提供情感类别标签。