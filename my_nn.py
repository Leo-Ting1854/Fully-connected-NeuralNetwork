
import numpy as np
import matplotlib.pyplot as plt
class Neural:
    def __init__(self):
        self.Weight=[]
        self.Bias=[]
        self.Modle={}
        self.Gradeint={}
        self.optimizer=""
    def model(self,input_dim,output_dim,hiden_layer,hiden_dim):
        hiden_dim.append(output_dim)
        hiden_dim.insert(0,input_dim)
        for lay in range(hiden_layer+1):#W層數為HIDEN層數+1
            w=np.random.rand(hiden_dim[lay],hiden_dim[lay+1])
            self.Weight.append(w)
            b=np.zeros((1,hiden_dim[lay+1]))
            self.Bias.append(b)
    def Forwardpass(self,X,Weight,Bias):

            layer=len(Weight)
            A=[]
            A_dev=[]
            Prob=[]
            for lay in range(layer):
                if(lay==0):
                    z=np.dot(X,Weight[lay])+Bias[lay]

                else:
                    z=np.dot(A[lay-1],Weight[lay])+Bias[lay]
                if(lay!=layer-1):#A只有LAYER-1層 最後一層是layer-2
                    A.append(np.tanh(z))
                    A_dev.append(1-np.power(A[lay],2))
                else:
                    up=np.exp(z)
                    down=np.sum(np.exp(z),axis=1,keepdims=True)
                    Prob=up/down
            Forward={'A':A,'A_dev':A_dev,"Prob":Prob}
            return Forward
    def Backwardpass(self,X,y,Weight,Forward):
        A,A_dev,Prob=Forward['A'],Forward['A_dev'],Forward['Prob']
        W_G=[]
        B_G=[]
        #求weight梯度  weight有len(Weight)層
        for lay in range(len(Weight)):
            if(lay==0):#最末層的W,G
                Pl_Pz=np.copy(Prob)
                Pl_Pz[range(0,len(y)),y]-=1
                d_W=np.dot(A[len(A)-1].T,Pl_Pz)
            else:
                #每一層Pl_Pz都是上一層的Pl_Pz,W.T相乘A導數
                Pl_Pz=np.dot(pl_pz_last,Weight[len(Weight)-lay].T)*A_dev[len(Weight)-1-lay]

                if(lay!=len(Weight)-1):#pz_pw=A.T
                    d_W=np.dot(A[len(A)-1-lay].T,Pl_Pz)
                else:#pz_pw=X.T
                    d_W=np.dot(X.T,Pl_Pz)
            W_G.append(d_W)
            B_G.append(np.sum(Pl_Pz,axis=0,keepdims=True))
            pl_pz_last=np.copy(Pl_Pz)#必須用copy
        #LIST 反轉成正向
        W_G.reverse()  #list.reverse沒有返回值 但會改變list順序
        B_G.reverse()
        return W_G,B_G
    def fit(self,X,y,iteration_time,learning_rate):
        Weight=self.Weight#完全相等  不是複製
        Bias=self.Bias
        lr_W=list.copy(Weight)
        lr_B=list.copy(Bias)
        #initial
        for lay in range(len(Weight)):
            lr_W[lay]=lr_W[lay]*0
            lr_B[lay]=lr_B[lay]*0
        #plot loss
        l=[]
        t=[]
        for time in range(iteration_time):
            Forward=self.Forwardpass(X,Weight,Bias)
            W_G,B_G=self.Backwardpass(X,y,Weight,Forward)
            for lay in range(len(Weight)):
                lr_B[lay]=lr_B[lay]+B_G[lay]**2
                lr_W[lay]=lr_W[lay]+W_G[lay]**2
            for lay in range(len(Weight)):
                Weight[lay]=Weight[lay]-learning_rate/(np.sqrt(lr_W[lay])+10**-8)*W_G[lay]
                Bias[lay]=Bias[lay]-learning_rate/(np.sqrt(lr_B[lay])+10**-8)*B_G[lay]
                # Weight[lay]=Weight[lay]-learning_rate*W_G[lay]#   lr設定0.001
                # Bias[lay]=Bias[lay]-learning_rate*B_G[lay]
            if(time%100==0):
                loss=self.Loss_caculator(y,Forward['Prob'])
                l.append(loss)
                t.append(time)
        plt.figure()
        plt.plot(t,l)
        plt.title('Loss')
        plt.xlabel('time')
        plt.ylabel('loss')

        self.Gradeint={'W_G':W_G,'B_G':B_G}
        self.Model={'Weight':Weight,'Bias':Bias}
    def predict(self,X):
        Weight=self.Model['Weight']
        Bias=self.Model['Bias']
        Forward=self.Forwardpass(X,Weight,Bias)
        prob=Forward['Prob']
        return np.argmax(prob,axis=1)
    def accuaracy(self,y_predict,y_test):
        same=0
        for data in range(len(y_predict)):
            if y_predict[data]==y_test[data]:
                same+=1
        return same/len(y_predict)*100
    def show_classification(self,Xtest,y):
        import matplotlib.pyplot as plt
        plt.figure()
        #only can do the binary_feature  classify
        #用X_test的 第一特徵當x軸  第二特徵當y軸
        plt.scatter(Xtest[:, 0], Xtest[:, 1], c=y, cmap=plt.cm.Spectral)
        x_min,x_max=Xtest[:,0].min(),Xtest[:,0].max()
        y_min,y_max=Xtest[:,1].min(),Xtest[:,1].max()
        X_point=np.linspace(x_min-0.5,x_max+0.5,100)
        Y_point=np.linspace(y_min-0.5,y_max+0.5,100)
        X,Y=np.meshgrid(X_point,Y_point)#座標點
        #然後我們要把這些X,Y 照著座標組合成一筆新的X_test
        #ravel可以很好的幫我把兩個座標點按照順序轉成1維
        #然後我就可以np.c_[] 重疊組合 生成新的X_TEST=(x,y) 2d點
        X_test=np.c_[X.ravel(),Y.ravel()]
        Z=self.predict(X_test)#但此時還是2維組成的序列
        Z=Z.reshape(X.shape)
        plt.contour(X,Y,Z)

    def Loss_caculator(self,y_real,y_prob):
        loss,cross_entropy=0,0
        cross_entropy=np.sum(-np.log(y_prob[range(0,len(y_prob)),y_real]))
        loss+=cross_entropy
        print('loss=',loss/len(y_real))#每一筆資料的loss

        return  loss/len(y_real)

def main():
    from sklearn import  datasets
    from sklearn.model_selection import  train_test_split
    from sklearn.preprocessing import  scale
    ##iris
    # iris=datasets.load_iris()
    # X=iris.data
    # y=iris.target
    #moon
    # X,y=datasets.make_moons(200,noise=0.2)

    #Circle
    # X,y=datasets.make_circles(200)

    #blobs
    # X,y=datasets.make_blobs(100)


    digit=datasets.load_digits()
    X=digit.data
    y=digit.target
    X=scale(X)
    # print(X.shape)
    # print(np.unique(y))


    x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.3)

    input_dim=X.shape[1]
    output_dim=len(np.unique(y))

    network=Neural()
    network.model(input_dim,output_dim,2,[20,20])
    network.fit(x_train,y_train,20000,1)

    y_p=network.predict(x_train)
    grade=network.accuaracy(y_p,y_train)
    print('在訓練資料上的準確度=',grade)

    y_p=network.predict(x_test)
    grade=network.accuaracy(y_p,y_test)
    print('在測試資料上的準確度=',grade)
    if(input_dim==2):
        network.show_classification(X,y)
    plt.show()
if __name__ == '__main__':
    main()
