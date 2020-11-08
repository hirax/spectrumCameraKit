import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

#　colour-science を
# 　PiPi からインストールしておく
from colour.plotting import *

#----------UI-----------------------
#import pyto_ui as ui
#view = ui.View()
#view.background_color = ui.COLOR_SYSTEM_BACKGROUND
#ui.show_view(view, ui.PRESENTATION_MODE_SHEET) # to show the view:
#----------UI-----------------------

#-----------設定-----------------------
# どのような処理を行うかを設定する
#-------------------------------------

# 動画をファイルから読み込むか、カメラから読み込むかを決める
isCamera = True

frameRate = 25 # frame/second
captureDurationInSecond = 20 # 撮影時間(sec.)

# デバイスが撮影するだろう画像サイズ（記入値はiPhone 11 の場合）
assumptionW = 480 # 480 #1920
assumptionH = 360 # 360  #1080

# （計算時間短縮のために）撮影画像の長さを何分の一にするか
resizeRatioInLength = 1  # 10分の1
aW = int( assumptionW / resizeRatioInLength )
aH = int( assumptionH / resizeRatioInLength )

# デバイスが撮影するだろう画像サイズ下で、
# 分光情報が得られるだろうX位置・領域 (これも想定)
# 0-1で指定
spectorPixelStart = 0.6
spectorPixelWidth = 0.3
aSpectorPixelStart = int( aH * spectorPixelStart)
aSpectorPixelWidth = int( aH * spectorPixelWidth)

# ------------撮影時の歪み補正用の行列を生成する------------
# 画像サイズや歪み程度が一定なら、動画読み込みより事前に処理しておく
#m1 = 0.1 # ずれ補正の傾斜調整、1 より小さな値に設定する、0になると補正量は0になる
m2 = int(-40.0 / resizeRatioInLength) #  中央と上下両端での、横方向ズレをピクセルで表したもの
mapY = np.zeros( (aH, aW), dtype=np.float32 ) 
mapX = np.zeros( (aH, aW), dtype=np.float32 )

for x in range( aW ): 
    mapX[:, x] = x # X方向は変化させない
for y in range( aH ): 
    for x in range( aW ): 
        mapY[y, x] = y + m2 * math.cos( (float(x)-float(aW)/2.0) / (float(aW)/2.0) * math.pi/2.0 )

cv2.imwrite( 'mapY.png', mapY )
# ------------撮影時の歪み補正用の行列を生成する------------

#-----------動画デバイスを開く(撮影ループ)-----------------------
frames = []
if isCamera: # カメラから動画を読み込む場合
    cap = cv2.VideoCapture( 0 ) # 0:back, 1: front
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'));
    #cap.set( cv2.CAP_PROP_FPS, frameRate ) 
    ret = cap.set( cv2.CAP_PROP_FRAME_WIDTH, assumptionW )
    ret = cap.set( cv2.CAP_PROP_FRAME_HEIGHT, assumptionH ) 
    # 画像サイズを取得する
    w = round( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    h = round( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    #print( w )
    #print( h )
    # カメラ時は「読み込む時間長さ（フレーム数）」を決めておく
    frame_n = frameRate * captureDurationInSecond
else:        # ファイルから動画を読み込む場合
    cap = cv2.VideoCapture( "sample.MOV" )
    frame_n = round( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
if not cap.isOpened(): # 動画ファイルを開くことができなかったら
    #print("VideoCapture can't be opened.")
    exit()
#---------------------------------------------------------

#-------------------動画読み取りループ-----------------------
n = 0
while( True ):
    ret, frame = cap.read()  # 動画象読み取り
    # 各種終了処理
    if not ret: # 画像を読み取れなかった場合
        #print("VideoCapture can't be read.")
        continue
    # Convert from BGR to RGB
    # BGRでなくて、RGBになってるので、入れ替える
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    frames.append(frame )
    # キャプチャー画像表示
    cv2.imshow('frame', frame)
    n = n+1
    if n >=  frame_n:
        break
cap.release() # 動画象読み取り終了
#-----------動画取得の終了-----------------------

#-----------動画加工処理-----------------------
undistortFrames = []
for frame in frames:
    if resizeRatioInLength != 1:
        frame = cv2.resize( frame, (aW, aH) )
    # 歪み除去画像を表示
    undistortFrame = cv2.remap(frame, mapX, mapY, cv2.INTER_CUBIC)
    undistortFrames.append( undistortFrame )
    cv2.imshow('frame', undistortFrame)
del frames  # メモリ削減のために撮影フレーム削除
#-----------動画加工処理-----------------------

print(undistortFrames[0].shape)

#-----------簡易ＲＧＢ画像出力-----------------------
# RGB3チャンネル画像格納用行列を作成する
# 単純化したRGB値をどの（x方向）画素から読み込むか
aBOffset = int( aSpectorPixelWidth / 6.0 )
aGOffset = int( 3 * aSpectorPixelWidth / 6.0 )
aROffset = int( 5 * aSpectorPixelWidth / 6.0 )

simpleRGBImg = np.zeros( (frame_n, aW, 3), np.float )
n = 0
for frame in undistortFrames:
    simpleRGBImg[ n, :, 2 ] = frame[ aSpectorPixelStart + aBOffset, :, 0 ].astype( np.float )*5 # B
    simpleRGBImg[ n, :, 1 ] = frame[ aSpectorPixelStart + aGOffset, :, 1 ].astype( np.float ) # G
    simpleRGBImg[ n, :, 0 ] = frame[ aSpectorPixelStart + aROffset, :, 2 ].astype( np.float )*5 # R
    n = n+1
#print(simpleRGBImg.shape)
#print(simpleRGBImg[0][0])
#print( np.max( simpleRGBImg ) )
#print( np.min( simpleRGBImg ) )

plt.figure(figsize=(3, 3), dpi=200)
plt.imshow(simpleRGBImg/255.)
plt.show()


