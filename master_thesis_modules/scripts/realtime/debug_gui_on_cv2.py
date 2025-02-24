import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_japanese_text(img, text, position, text_color=(255, 255, 255), bg_color=None, font_path="NotoSansCJK-Regular.ttc", font_size=30):
    """
    OpenCVの画像に日本語テキストを描画する関数

    Parameters:
        img (numpy.ndarray): OpenCVで読み込んだ画像
        text (str): 描画する日本語テキスト
        position (tuple): テキストの左上座標 (x, y)
        text_color (tuple): 文字色（BGR）
        bg_color (tuple or None): 背景色（BGR）、Noneの場合は背景なし
        font_path (str): 使用するフォントのパス（日本語対応フォント）
        font_size (int): フォントサイズ

    Returns:
        numpy.ndarray: 日本語テキストを描画した画像
    """
    # OpenCVの画像をPillow形式に変換
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # フォントを設定
    font = ImageFont.truetype(font_path, font_size)

    # 背景色が指定されていれば背景を描画
    if bg_color:
        text_size = draw.textbbox(position, text, font=font)  # (left, top, right, bottom)
        draw.rectangle(text_size, fill=bg_color)

    # 日本語テキストを描画
    draw.text(position, text, font=font, fill=text_color)

    # Pillowの画像をOpenCV形式に変換して返す
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 使用例
if __name__ == "__main__":
    # jpg_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250224PlayChime/jpg/bbox/l_1724752846.853.jpg"
    jpg_path="/catkin_ws/src/database/20250224PlayChime/jpg/bbox/l_1724752846.853.jpg"
    img = cv2.imread(jpg_path)  # 画像を読み込む
    output_img = draw_japanese_text(img, "こんにちは", (50, 50), text_color=(255, 255, 255), bg_color=(0, 0, 0), font_path="NotoSansCJK-Regular.ttc")
    cv2.imshow("Result", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
