
'''
古彝文生成及修复
主程序 查看损失值
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    with open('train_losses.pkl', 'rb') as f:
        losses = pickle.load(f)
        f.close()
        
        fig, ax = plt.subplots(figsize=(16,8))
        losses = np.array(losses)
        #plt.plot(losses.T[0]+losses.T[1], label='Discriminator Total Loss')
        plt.plot(losses.T[0], label='Discriminator Loss')
        plt.plot(losses.T[1], label='Generator Loss')
        plt.title("Training Losses")
        plt.legend()
        plt.show()
    