import os
import shutil

def clean_training_files():
    # 清理tmp文件夹
    tmp_dir = 'tmp/maddpg_3uav_5target/'
    if os.path.exists(tmp_dir):
        print(f"正在删除 {tmp_dir}...")
        shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"已重新创建空文件夹 {tmp_dir}")
    
    # 清理图片文件夹
    images_dir = 'images/'
    if os.path.exists(images_dir):
        print(f"正在删除 {images_dir}...")
        shutil.rmtree(images_dir)
        os.makedirs(images_dir, exist_ok=True)
        print(f"已重新创建空文件夹 {images_dir}")
    
    # 删除得分历史文件
    score_file = 'score_history_3uav_5target.csv'
    if os.path.exists(score_file):
        print(f"正在删除 {score_file}...")
        os.remove(score_file)
        print(f"已删除 {score_file}")

if __name__ == '__main__':
    print("开始清理训练文件...")
    clean_training_files()
    print("清理完成!")