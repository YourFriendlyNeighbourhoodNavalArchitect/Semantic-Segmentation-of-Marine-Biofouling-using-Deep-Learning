from trainModel import trainAttentionUNet
from trainSImpleUNet import trainSimpleUNet

if __name__ == "__main__":
    print("Starting Attention U-Net training...")
    trainAttentionUNet()
    print("\nStarting Simple U-Net training...")
    trainSimpleUNet()
