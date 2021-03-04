import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#csvのディレクトリを指定
#df_cu0 = pd.read_csv('log/mobile-curl_sac-01-06-07:16:55-im84-b256-s1-pixel/csv/log.csv')
#df_cu1 = pd.read_csv('log/mobile-curl_sac-01-29-03:58:01-im84-b256-s123-pixel/csv/log.csv')
#df_cu2 = pd.read_csv('log/mobile-curl_sac-01-30-03:49:16-im84-b256-s21-pixel/csv/log.csv')
#df_cu3 = pd.read_csv('log/mobile-curl_sac-02-02-08:31:10-im84-b256-s8-pixel/csv/log.csv')
#df_cu4 = pd.read_csv('log/mobile-curl_sac-02-03-13:07:22-im84-b256-s666-pixel/csv/log.csv')

df_cu0 = pd.read_csv('log/kuka-curl_sac-01-10-13:49:21-im84-b256-s1-pixel/csv/log.csv')
df_cu1 = pd.read_csv('log/kuka-curl_sac-02-09-07:34:49-im84-b256-s123-pixel/csv/log.csv')
df_cu2 = pd.read_csv('log/kuka-curl_sac-02-11-12:54:15-im84-b256-s21-pixel/csv/log.csv')
df_cu3 = pd.read_csv('log/kuka-curl_sac-02-21-23:11:47-im84-b256-s8-pixel/csv/log.csv')


df_cus = {}
df_cus[0] = df_cu0
df_cus[1] = df_cu1
df_cus[2] = df_cu2
df_cus[3] = df_cu3
#df_cus[4] = df_cu4

panel_cu = pd.Panel(df_cus)
df_cu_mean = panel_cu.mean(axis=0)
df_cu_std = panel_cu.std(axis=0)

#csvのディレクトリを指定
#df_ae0 = pd.read_csv('log/mobile-sac_ae-01-07-07:25:27-im84-b256-s1-pixel/csv/log.csv')
#df_ae1 = pd.read_csv('log/mobile-sac_ae-02-05-03:02:18-im84-b256-s123-pixel/csv/log.csv')
#df_ae2 = pd.read_csv('log/mobile-sac_ae-02-05-23:50:08-im84-b256-s21-pixel/csv/log.csv')
#df_ae3 = pd.read_csv('log/mobile-sac_ae-02-06-15:10:13-im84-b256-s8-pixel/csv/log.csv')
#df_ae4 = pd.read_csv('log/mobile-sac_ae-02-07-06:29:00-im84-b256-s666-pixel/csv/log.csv')

df_ae0 = pd.read_csv('log/kuka-sac_ae-01-12-04:04:09-im84-b256-s1-pixel/csv/log.csv')
df_ae1 = pd.read_csv('log/kuka-sac_ae-02-14-00:37:35-im84-b256-s123-pixel/csv/log.csv')
df_ae2 = pd.read_csv('log/kuka-sac_ae-02-15-23:27:24-im84-b256-s21-pixel/csv/log.csv')
df_ae3 = pd.read_csv('log/kuka-sac_ae-02-20-01:01:09-im84-b256-s8-pixel/csv/log.csv')

df_aes = {}
df_aes[0] = df_ae0
df_aes[1] = df_ae1
df_aes[2] = df_ae2
df_aes[3] = df_ae3
#df_aes[4] = df_ae4

panel_ae = pd.Panel(df_aes)
df_ae_mean = panel_ae.mean(axis=0)
df_ae_std = panel_ae.std(axis=0)

#csvのディレクトリを指定
#df_gt0 = pd.read_csv('log/mobile-curl_sac-01-14-08:43:33-im84-b256-s1-identity/csv/log.csv')
#df_gt1 = pd.read_csv('log/mobile-curl_sac-02-08-01:46:56-im84-b256-s123-identity/csv/log.csv')
#df_gt2 = pd.read_csv('log/mobile-curl_sac-02-08-08:43:57-im84-b256-s21-identity/csv/log.csv')
#df_gt3 = pd.read_csv('log/mobile-curl_sac-02-08-15:48:26-im84-b256-s8-identity/csv/log.csv')
#df_gt4 = pd.read_csv('log/mobile-curl_sac-02-08-23:59:57-im84-b256-s666-identity/csv/log.csv')

df_gt0 = pd.read_csv('log/kuka-curl_sac-01-14-15:21:51-im84-b256-s1-identity/csv/log.csv')
df_gt1 = pd.read_csv('log/kuka-curl_sac-02-17-22:20:49-im84-b256-s123-identity/csv/log.csv')
df_gt2 = pd.read_csv('log/kuka-curl_sac-02-19-03:07:30-im84-b256-s21-identity/csv/log.csv')
df_gt3 = pd.read_csv('log/kuka-curl_sac-02-24-02:13:24-im84-b256-s8-identity/csv/log.csv')


df_gts = {}
df_gts[0] = df_gt0
df_gts[1] = df_gt1
df_gts[2] = df_gt2
df_gts[3] = df_gt3
#df_gts[4] = df_gt4

panel_gt = pd.Panel(df_gts)
df_gt_mean = panel_gt.mean(axis=0)
df_gt_std = panel_gt.std(axis=0)

X_cu = df_cu0['step'].values
y_cu = df_cu_mean['mean_distance_to_goal'].values
std_cu = df_cu_std['mean_distance_to_goal'].values

X_ae = df_ae0['step'].values
y_ae = df_ae_mean['mean_distance_to_goal'].values
std_ae = df_ae_std['mean_distance_to_goal'].values

X_gt = df_gt0['step'].values
y_gt = df_gt_mean['mean_distance_to_goal'].values
std_gt = df_gt_std['mean_distance_to_goal'].values

plt.rcParams['font.size'] = 10
plt.rcParams["mathtext.fontset"] = "stix" # stixフォントにする

plt.rcParams["xtick.direction"] = "in" #X軸の目盛の向き
plt.rcParams["ytick.direction"] = "in" #Y軸の目盛の向き
plt.rcParams["xtick.major.width"] = 1.5 #X軸の主目盛の太さ
plt.rcParams["ytick.major.width"] = 1.5 #Y軸の主目盛の太さ
plt.rcParams["xtick.minor.width"] = 1.2 #X軸の副目盛の太さ
plt.rcParams["ytick.minor.width"] = 1.2 #Y軸の副目盛の太さ
plt.rcParams["xtick.major.size"] = 4.5 #X軸の主目盛の長さ
plt.rcParams["ytick.major.size"] = 4.5 #Y軸の主目盛の長さ
plt.rcParams["xtick.minor.size"] = 3.0 #X軸の副目盛の長さ
plt.rcParams["ytick.minor.size"] = 3.0 #Y軸の副目盛の長さ
plt.rcParams["xtick.labelsize"] = 10.0 #X軸の目盛りラベルのフォントサイズ
plt.rcParams["ytick.labelsize"] = 10.0 #Y軸の目盛ラベルのフォントサイズ
plt.rcParams["xtick.major.pad"] = 5 #X軸と目盛ラベルの間隔(単位はポイント)
plt.rcParams["ytick.major.pad"] = 5  #Y軸と目盛ラベルの間隔(単位はポイント)
plt.rcParams["axes.labelsize"] = 10   #図のフォントラベルサイズ
plt.rcParams["axes.linewidth"] = 1.5  #図の枠線の太さ
plt.rcParams["axes.labelpad"] = 6 #軸と軸ラベルの間隔



plt.plot(X_cu, y_cu, color='tab:orange', label='CL (Ours)', lw=1.5)
plt.plot(X_ae, y_ae, color='tab:blue', label='VAE', lw=1.5)
plt.plot(X_gt, y_gt, color='tab:green', label='Ground Truth', lw=1.5)


plt.fill_between(X_ae,
                 y_ae + std_ae,
                 y_ae - std_ae,
                 alpha=0.15, color='tab:blue')



plt.fill_between(X_cu,
                 y_cu + std_cu,
                 y_cu - std_cu,
                 alpha=0.15, color='tab:orange')


plt.fill_between(X_gt,
                 y_gt + std_gt,
                 y_gt - std_gt,
                 alpha=0.15, color='tab:green')





plt.grid()
plt.legend(loc='upper right')
#plt.legend(loc='lower left')
plt.xlabel('steps')
plt.ylabel('Distance to Goal [m]')
#plt.title('Mobile Navigation Learning Curve')
plt.title('Robot Arm Learning Curve')
#plt.ylim([0.0, 0.45])
plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="x")
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))#y軸小数点以下3桁表示
plt.locator_params(axis='y',nbins=10)#y軸，6個以内．
plt.tight_layout()
plt.savefig("result/result.svg")
plt.show()
