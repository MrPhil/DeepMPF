import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Lambda, GaussianNoise, BatchNormalization, Reshape, dot, Activation, concatenate, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.engine.topology import Layer
from keras.utils import plot_model
from keras.datasets import mnist
from keras import backend as K
from random import shuffle
from keras.callbacks import ReduceLROnPlateau
import csv
csv.field_size_limit(500 * 1024 * 1024)
import numpy as np
import math
import datetime
from pyhht.emd import EMD
from scipy import fftpack
from numpy import linalg as la

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):  #
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 1
        while counter < len(row):
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):
    SampleFeature = []

    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        PairFeature.append(SequenceList[counter][0])

        FeatureMatrix = []
        counter1 = 0
        while counter1 < PaddingLength:
            row = []
            counter2 = 0
            while counter2 < len(EmbeddingList[0]) - 1:
                row.append(0)
                counter2 = counter2 + 1
            FeatureMatrix.append(row)
            counter1 = counter1 + 1

        try:
            counter3 = 0
            while counter3 < PaddingLength:
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
                    counter4 = counter4 + 1
                counter3 = counter3 + 1
        except:
            pass

        PairFeature.append(FeatureMatrix)
        SampleFeature.append(PairFeature)
        counter = counter + 1
    return SampleFeature

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)
        for c in g_members:
            result[c] = str(tar_list[index])
        index = index + 1
    return result

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=int(n/base)
        ch1=chars[n%base]
        n=int(n/base)
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com

def get_4_nucleotide_composition(tris, seq, pythoncount=True,protein=0):
    seq_len = len(seq)
    tri_feature = [0] * len(tris)
    k = len(tris[0])
    note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                note_feature[ind][x] = note_feature[ind][x] + 1
        try:
            u, s, v = la.svd(note_feature)
            for i in range(len(s)):
                tri_feature = tri_feature + u[i] * s[i] / seq_len
        except:
            print(protein)
    return fftpack.hilbert(tri_feature)

def translate_sequence (seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    #TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def GenerateEmbeddingFeatureSVD(SequenceList):   #

    SampleFeature = []
    protein_tris = get_3_protein_trids()
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    for i in range(len(SequenceList)):
        if i % 100==0:
            print(str(i)+'/'+str(len(SequenceList)))
        protein_seq = translate_sequence(SequenceList[i][1], group_dict)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount=False, protein=SequenceList[i][0])
        row=[]
        row.append(SequenceList[i][0])
        row.append(protein_tri_fea)
        SampleFeature.append(row)

    return SampleFeature

def GenerateSampleFeature(InteractionList, EmbeddingFeature1, EmbeddingFeature2,useword2vecforPro):
    SampleFeature1 = []
    SampleFeature2 = []

    counter = 0
    while counter < len(InteractionList):
        if counter%50==0:
            print(counter,'/',len(InteractionList))
        Pair1 = InteractionList[counter][0]
        Pair2 = InteractionList[counter][1]

        counter1 = 0
        while counter1 < len(EmbeddingFeature1):
            if EmbeddingFeature1[counter1][0] == Pair1:
                SampleFeature1.append(EmbeddingFeature1[counter1][1])
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(EmbeddingFeature2):
            if EmbeddingFeature2[counter2][0] == Pair2:
                SampleFeature2.append(EmbeddingFeature2[counter2][1])
                break
            counter2 = counter2 + 1

        counter = counter + 1

    SampleFeature1 = np.array(SampleFeature1)
    SampleFeature2 = np.array(SampleFeature2)

    if useword2vecforPro == 'w2v_Pro':
        SampleFeature1 = SampleFeature1.reshape(SampleFeature1.shape[0], SampleFeature1.shape[1], SampleFeature1.shape[2], 1)
    SampleFeature2 = SampleFeature2.reshape(SampleFeature2.shape[0], SampleFeature2.shape[1], SampleFeature2.shape[2], 1)
    return SampleFeature1, SampleFeature2

def GenerateBehaviorFeature(InteractionPair, metapath_target, metapath_drug, net): #
    SampleFeature1 = []
    SampleFeature2 = []
    k = []
    row = []
    for i in range(len(metapath_target[0][1:])):
        row.append(0)
    for i in range(len(InteractionPair)):
        if i%2000 == 0:
            print(i)
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]
        flag1=0
        flag2=0
        for m in range(len(metapath_target)):
            if Pair1 == metapath_target[m][0]:
                SampleFeature1.append(metapath_target[m][1:])
                flag1=1
                break
        if flag1==0:
            SampleFeature1.append(row)

        for n in range(len(metapath_drug)):
            if Pair2 == metapath_drug[n][0]:
                SampleFeature2.append(metapath_drug[n][1:])
                flag2=1
                break
        if flag2==0:
            SampleFeature2.append(row)

    SampleFeature1 = np.array(SampleFeature1)
    SampleFeature2 = np.array(SampleFeature2)
    return SampleFeature1, SampleFeature2

def GenerateBehaviorFeaturesimi(InteractionPair, metapath_target, metapath_drug, net,pro_l,drug_l): #
    SampleFeature1 = []
    SampleFeature2 = []
    k = []
    row = []
    for i in range(len(metapath_target[0][1:])):
        row.append('0')

    row1 = []
    for i in range(len(metapath_drug[0][1:])):
        row1.append('0')
    for i in range(len(InteractionPair)):
        if i%2000 == 0:
            print(i)
        Pair1 = int(InteractionPair[i][0])
        Pair2 = int(InteractionPair[i][1])
        flag1=0
        flag2=0
        tempt = pro_l[Pair1][0][:3]+pro_l[Pair1][0][4:]
        for i in range(len(metapath_target)):
            if tempt == metapath_target[i][0]:
                SampleFeature1.append(metapath_target[i][1:])
                flag1=1
                break
        if flag1==0:
            SampleFeature1.append(row)
            print('1error'+str(Pair1))

        for i in range(len(metapath_drug)):
            if drug_l[Pair2][0] == metapath_drug[i][0]:
                SampleFeature2.append(metapath_drug[i][1:])
                flag2=1
                break
        if flag2==0:
            SampleFeature2.append(row1)
            print('2error'+str(Pair2))


    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')
    # k = np.array(k).astype('float32')
    # return SampleFeature1, SampleFeature2,k
    # SampleFeature1=GenerHilbertFeature(SampleFeature1)
    # SampleFeature2 = GenerHilbertFeature(SampleFeature2)
    return SampleFeature1, SampleFeature2

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label


if __name__ == '__main__':
    datasetNum=3
    foldd=0
    while foldd<5:
        useword2vecforPro='SVD_Pro'  # 0=SVD_Pro   1=w2v_Pro
        representation = 'representation/'  # 16d 32d 128d
        useCNN=0  # 1=useCNN
        drawN = str(foldd)+'/'  # storeName    #####################2
        datasetNameL=['Enzyme', 'GPCR', 'Ion channel', 'Nuclear receptor']  #
        datasetName = datasetNameL[datasetNum]
        batch_sizeL=[128,4,16,2]
        batch_size = batch_sizeL[datasetNum]
        feature_method='metaPath'  # metaPath    meta     magnn  line  deepWalk

        print(datasetName+','+useword2vecforPro+','+feature_method+','+representation[:-1]+',useCNN='+str(useCNN))
        nameF = '../fiveFold/'+useword2vecforPro+'/'+feature_method+'/'+datasetName+'/' + drawN

        AllDrugCanonicalSMILES = []
        ReadMyCsv1(AllDrugCanonicalSMILES, representation+datasetName+'_index_smileFinish.csv')  # 1940
        DrugEmbedding = []
        ReadMyCsv3(DrugEmbedding, representation+'DrugEmbedding343d.csv')  #

        PositiveSample_Train = []
        ReadMyCsv1(PositiveSample_Train, nameF + 'rPositiveSample_Train_ddi.csv')  # 153472
        PositiveSample_Validation = []
        ReadMyCsv1(PositiveSample_Validation, nameF + 'rPositiveSample_Validation_ddi.csv')  # 21925
        PositiveSample_Test = []
        ReadMyCsv1(PositiveSample_Test, nameF + 'rPositiveSample_Test_ddi.csv')  # 43850

        NegativeSample_Train = []
        ReadMyCsv1(NegativeSample_Train, nameF + 'rNegativeSample_Train_ddi.csv')
        NegativeSample_Validation = []
        ReadMyCsv1(NegativeSample_Validation, nameF + 'rNegativeSample_Validation_ddi.csv')
        NegativeSample_Test = []
        ReadMyCsv1(NegativeSample_Test, nameF + 'rNegativeSample_Test_ddi.csv')

        x_train_pair = []
        x_train_pair.extend(PositiveSample_Train)
        x_train_pair.extend(NegativeSample_Train)

        x_validation_pair = []
        x_validation_pair.extend(PositiveSample_Validation)
        x_validation_pair.extend(NegativeSample_Validation)

        x_test_pair = []
        x_test_pair.extend(PositiveSample_Test)
        x_test_pair.extend(NegativeSample_Test)

        AllProCanonicalSMILES = []
        ReadMyCsv1(AllProCanonicalSMILES, representation + datasetName + '_index_sequenceFinish.csv')  # 1940

        if useword2vecforPro == 'SVD_Pro':
            ProEmbeddingFeature = GenerateEmbeddingFeatureSVD(AllProCanonicalSMILES)


        DrugEmbeddingFeature = GenerateEmbeddingFeature(AllDrugCanonicalSMILES, DrugEmbedding, 64)


        if feature_method=='meta':
            metapath_target = []
            ReadMyCsv1(metapath_target, representation+'meta'+drawN+datasetName+'_target_embedding.csv')
            metapath_drug = []
            ReadMyCsv1(metapath_drug, representation+'meta'+drawN+datasetName+'_drug_embedding.csv')
        elif feature_method=='magnn':
            metapath_target = []
            ReadMyCsv1(metapath_target, representation + 'MAGNN' + drawN + datasetName + '_target_embedding.csv')
            metapath_drug = []
            ReadMyCsv1(metapath_drug, representation + 'MAGNN' + drawN + datasetName + '_drug_embedding.csv')
        elif feature_method=='metaPath':
            metapath_target = []
            ReadMyCsv1(metapath_target, representation + 'metaPath' + drawN + datasetName + '_target_embedding.csv')
            metapath_drug = []
            ReadMyCsv1(metapath_drug, representation + 'metaPath' + drawN + datasetName + '_drug_embedding.csv')
        elif feature_method == 'line':
            metapath_target = []
            if datasetName == 'Ion channel':
                tttt = 'Ion_channel'
            elif datasetName == 'Nuclear receptor':
                tttt = 'Nuclear_receptor'
            else:
                tttt = datasetName
            ReadMyCsv1(metapath_target,
                       '../2extract_feature/behavior/OpenNE-master/' + tttt + '/' + str(
                           foldd) + '/' + tttt + '_line64d.csv')
            metapath_drug = []
            ReadMyCsv1(metapath_drug,
                       '../2extract_feature/behavior/OpenNE-master/' + tttt + '/' + str(
                           foldd) + '/' + tttt + '_line64d.csv')
        elif feature_method == 'deepWalk':
            metapath_target = []
            if datasetName == 'Ion channel':
                tttt = 'Ion_channel'
            elif datasetName == 'Nuclear receptor':
                tttt = 'Nuclear_receptor'
            else:
                tttt = datasetName
            ReadMyCsv1(metapath_target,
                       '../2extract_feature/behavior/OpenNE-master/' + tttt + '/' + str(
                           foldd) + '/' + tttt + '_deepWalk64d.csv')
            metapath_drug = []
            ReadMyCsv1(metapath_drug,
                       '../2extract_feature/behavior/OpenNE-master/' + tttt + '/' + str(
                           foldd) + '/' + tttt + '_deepWalk64d.csv')

        input_file = open('representation/Protein sequence similarity matrix of the gold standard/'+datasetName+'.txt', 'r')
        input_file.readline() # skip first line
        simiPro=[]
        for line in input_file:
            row=line.strip().split('\t')
            simiPro.append(row)
        input_file.close()

        input_file = open(
            'representation/Compound structure similarity matrix of the gold standard/' + datasetName + '.txt', 'r')
        input_file.readline()  # skip first line
        simiDrug = []
        for line in input_file:
            row = line.strip().split('\t')
            simiDrug.append(row)
        input_file.close()


        def mean_drug(seq):
            save = []
            # tri_feature = [0] * seq.shape[2]
            seqq = seq.reshape(seq.shape[0], seq.shape[1], seq.shape[2])  # n*len(smile)*embedding_num
            for i in range(seqq.shape[0]):
                try:
                    tri_feature = seqq[i].mean(axis=0)
                except:
                    print(i)
                save.append(tri_feature)
            save = np.array(save)
            return save


        x_train_1_Attribute, x_train_2_Attribute = GenerateSampleFeature(x_train_pair, ProEmbeddingFeature, DrugEmbeddingFeature,useword2vecforPro)  # drug and miRNA feature. matrix and vector
        x_validation_1_Attribute, x_validation_2_Attribute = GenerateSampleFeature(x_validation_pair, ProEmbeddingFeature, DrugEmbeddingFeature,useword2vecforPro)
        x_test_1_Attribute, x_test_2_Attribute = GenerateSampleFeature(x_test_pair, ProEmbeddingFeature, DrugEmbeddingFeature,useword2vecforPro)


        if useCNN!=1:
            x_train_2_Attribute = mean_drug(x_train_2_Attribute)
            x_test_2_Attribute = mean_drug(x_test_2_Attribute)
            x_validation_2_Attribute = mean_drug(x_validation_2_Attribute)

        if useword2vecforPro == 'w2v_Pro' and useCNN!=1:
            x_train_1_Attribute = mean_drug(x_train_1_Attribute)
            x_test_1_Attribute = mean_drug(x_test_1_Attribute)
            x_validation_1_Attribute = mean_drug(x_validation_1_Attribute)


        DDI_net = []
        ReadMyCsv1(DDI_net, '../0dataset/'+datasetName+'/TDI.csv')

        x_train_1_KG, x_train_2_KG = GenerateBehaviorFeature(x_train_pair, metapath_target, metapath_drug, DDI_net)
        x_validation_1_KG, x_validation_2_KG = GenerateBehaviorFeature(x_validation_pair, metapath_target, metapath_drug, DDI_net)
        x_test_1_KG, x_test_2_KG = GenerateBehaviorFeature(x_test_pair, metapath_target, metapath_drug, DDI_net)


        DDI_net = []
        ReadMyCsv1(DDI_net, '../0dataset/'+datasetName+'/TDI.csv')
        pro_l = []
        ReadMyCsv1(pro_l, 'representation/Protein sequence similarity matrix of the gold standard/pr_list_' + datasetName +'.csv')
        drug_l = []
        ReadMyCsv1(drug_l, 'representation/Compound structure similarity matrix of the gold standard/dr_list_' + datasetName +'.csv')

        x_train_1_si, x_train_2_si = GenerateBehaviorFeaturesimi(x_train_pair, simiPro, simiDrug, DDI_net,pro_l,drug_l)
        x_validation_1_si, x_validation_2_si = GenerateBehaviorFeaturesimi(x_validation_pair, simiPro, simiDrug, DDI_net,pro_l,drug_l)
        x_test_1_si, x_test_2_si = GenerateBehaviorFeaturesimi(x_test_pair, simiPro, simiDrug, DDI_net,pro_l,drug_l)


        y_train_Pre = MyLabel(x_train_pair)     # Label->one hot
        y_validation_Pre = MyLabel(x_validation_pair)
        y_test_Pre = MyLabel(x_test_pair)
        num_classes = 2
        y_train = keras.utils.to_categorical(y_train_Pre, num_classes)
        y_validation = keras.utils.to_categorical(y_validation_Pre, num_classes)
        y_test = keras.utils.to_categorical(y_test_Pre, num_classes)


        print('x_train_1_Attribute shape', x_train_1_Attribute.shape)
        print('x_train_2_Attribute shape', x_train_2_Attribute.shape)
        print('x_train_1_KG shape', x_train_1_KG.shape)
        print('x_train_2_KG shape', x_train_2_KG.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

        #———————————————————— 5 times —————————————————————
        CounterT = 0
        starttime = datetime.datetime.now()
        while CounterT < 1:
            # ———————————————————— define ————————————————————
            if useCNN != 1:
                input1 = Input(shape=(len(x_train_1_Attribute[0]),), name='input1')
                x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input1)
                # x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x1)
                x1 = Dropout(rate=0.3)(x1)
                input2 = Input(shape=(len(x_train_2_Attribute[0]),), name='input2')
                x2 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input2)
                # x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x1)
                x2 = Dropout(rate=0.3)(x2)
            if useCNN == 1:
                input1 = Input(shape=(len(x_train_1_Attribute[0]), len(x_train_1_Attribute[0][0]), 1), name='input1')
                x1 = Conv2D(64, kernel_size=(8, 64), activation='relu', name='conv1')(input1)
                x1 = BatchNormalization()(x1)
                # ## x1 = MaxPooling2D(pool_size=(1, 1), name='pool1')(x1)
                # x1 = GlobalAveragePooling2D()(x1)  # NIN
                # x1 = Reshape((1, 64))(x1)
                # x1 = Self_Attention(64)(x1)
                # x1 = GlobalAveragePooling1D()(x1)
                # x1 = Reshape((64, 64))(input1)
                # x1 = Self_Attention(64)(x1)
                # x1 = GlobalAveragePooling1D()(x1)
                x1 = Flatten()(x1)
                x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x1)
                x1 = Dropout(rate=0.3)(x1)

                input2 = Input(shape=(len(x_train_2_Attribute[0]), len(x_train_2_Attribute[0][0]), 1), name='input2')
                x2 = Conv2D(64, kernel_size=(8, 64), activation='relu', name='conv2')(input2)
                x2 = BatchNormalization()(x2)
                x2 = Flatten()(x2)
                x2 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x2)
                x2 = Dropout(rate=0.3)(x2)

            input3 = Input(shape=(len(x_train_1_KG[0]),), name='input3')
            x3 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input3)
            # x3 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x3)
            x3 = Dropout(rate=0.3)(x3)

            input4 = Input(shape=(len(x_train_2_KG[0]),), name='input4')
            x4 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input4)
            # x4 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x4)
            x4 = Dropout(rate=0.3)(x4)

            # input3 = Input(shape=(len(train3[0]),), name='input3')
            # # x3 = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.001))(input3)
            # x3 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input3)
            # x3 = Dropout(rate=0.1)(x3)

            input5 = Input(shape=(len(x_train_1_si[0]),), name='input5')
            x5 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input5)
            # x5 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x5)
            x5 = Dropout(rate=0.3)(x5)

            input6 = Input(shape=(len(x_train_2_si[0]),), name='input6')
            x6 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input6)
            # x6 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x6)
            x6 = Dropout(rate=0.3)(x6)

            flatten = keras.layers.concatenate([x1, x2, x3, x4, x5, x6])
            hidden = Dense(128, activation='relu', name='hidden3', activity_regularizer=regularizers.l2(0.001))(flatten)
            hidden = Dropout(rate=0.1)(hidden)
            # hidden = Dense(64, activation='relu', name='hidden4', activity_regularizer=regularizers.l2(0.001))(hidden)
            # hidden = Dropout(rate=0.1)(hidden)
            hidden = Dense(32, activation='relu', name='hidden2', activity_regularizer=regularizers.l2(0.001))(hidden)
            hidden = Dropout(rate=0.1)(hidden)
            output = Dense(num_classes, activation='softmax', name='output')(hidden)  # category
            model = Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=output)
            model.summary()
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='auto')  # Automatically adjust the learning rate
            history = model.fit({'input1': x_train_1_Attribute, 'input2': x_train_2_Attribute,'input3': x_train_1_KG, 'input4': x_train_2_KG,
                                 'input5': x_train_1_si,
                                 'input6': x_train_2_si}, y_train,
                                validation_data=({'input1': x_validation_1_Attribute, 'input2': x_validation_2_Attribute,
                                                  'input3': x_validation_1_KG, 'input4': x_validation_2_KG,
                                                  'input5': x_validation_1_si, 'input6': x_validation_2_si
                                                  }, y_validation),
                                callbacks=[reduce_lr],
                                epochs=50, batch_size=batch_size,
                                )

            ModelTest = Model(inputs=model.input, outputs=model.get_layer('output').output)
            ModelTestOutput = ModelTest.predict(
                [x_test_1_Attribute, x_test_2_Attribute, x_test_1_KG, x_test_2_KG, x_test_1_si, x_test_2_si])

            print(ModelTestOutput.shape)
            print(type(ModelTestOutput))
            LabelPredictionProb = []
            LabelPrediction = []

            counter = 0
            while counter < len(ModelTestOutput):
                rowProb = []
                rowProb.append(y_test_Pre[counter])
                rowProb.append(ModelTestOutput[counter][1])
                LabelPredictionProb.append(rowProb)

                row = []
                row.append(y_test_Pre[counter])
                if ModelTestOutput[counter][1] > 0.5:
                    row.append(1)
                else:
                    row.append(0)
                LabelPrediction.append(row)

                counter = counter + 1
            if representation=='representation/':
                LabelPredictionProbName = './draw/' + useword2vecforPro + '/' +feature_method + '/' + datasetName + '/' + drawN + 'RealAndPredictionProbA+B' + str(CounterT) + '.csv'
                StorFile(LabelPredictionProb, LabelPredictionProbName)
                LabelPredictionName = './draw/' + useword2vecforPro + '/' +feature_method + '/' + datasetName + '/' + drawN + 'RealAndPredictionA+B' + str(CounterT) + '.csv'
                StorFile(LabelPrediction, LabelPredictionName)
                endtime = datetime.datetime.now()
                print(endtime - starttime)
            else:
                LabelPredictionProbName = representation+'draw/' + useword2vecforPro + '/' + datasetName + '/' + drawN + 'RealAndPredictionProbA+B' + str(
                    CounterT) + '.csv'
                StorFile(LabelPredictionProb, LabelPredictionProbName)
                LabelPredictionName = representation+'draw/' + useword2vecforPro + '/' + datasetName + '/' + drawN + 'RealAndPredictionA+B' + str(
                    CounterT) + '.csv'
                StorFile(LabelPrediction, LabelPredictionName)
                endtime = datetime.datetime.now()
                print(endtime - starttime)

            CounterT = CounterT + 1
        foldd=foldd+1

