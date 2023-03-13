from utils import decode_map_xml
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import json
import os

def load_data(name):
    with open('./'+name+'_test.json','r',encoding='utf-8') as reader:
        data = json.load(reader)
    print(f'data loaded:{name}')
    print(data[:-1])
    return data[-1]

def compare_xy_traj(scenario,data_dir):
    xml_dir = './scenarios/'+scenario+'/map.net.xml'
    plt = decode_map_xml(xml_dir,fig_size=(20,10),grid=True)
    
    fontsize=15
    plt.title('Urban Scenario: Double Merge',fontsize=20)
    plt.xlabel('x/m',fontsize=15)
    plt.ylabel('y/m',fontsize=15)
    
    # plt.xlim([-40,100])
    # plt.ylim([10,60])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)

    datas = load_data(data_dir)
    print(len(datas))
    for data in tqdm(datas):
        x,y = [d[0] for d in data],[d[1] for d in data]
        if y[-1]>54 and y[-1]<56:
            # plt.plot(x,y,color='blue')
            pass
        if y[-1]<20 and x[-1]>160:
            plt.scatter(x, y,color='steelblue',s=10)
    
    scenario = scenario.split('/')[-1]
    
    plt.savefig(f'./stat_pic/xy_{data_dir}_{scenario}.png')
    plt.savefig(f'./stat_pic/xy_{data_dir}_{scenario}.svg')

def compare_xy_roundabout(dir_list,name='map_glb'):
    xml_dir = './scenarios/roundabout/map.net.xml'
    plt = decode_map_xml(xml_dir,fig_size=(10,10),grid=True)
    fontsize=15
    plt.title('Urban Scenario: Roundabout-C',fontsize=20)
    plt.xlabel('x/m',fontsize=15)
    plt.ylabel('y/m',fontsize=15)
    
    # plt.xlim([60,150])
    # plt.ylim([60,150])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)
    # c_list=['skyblue','steelblue','midnightblue']
    for i,data_dir in enumerate(dir_list):
        datas = load_data(data_dir)
        print(len(datas))
        for data in tqdm(datas):
            x,y = [d[0] for d in data],[d[1] for d in data]
            # if y[-1]<20 and x[-1]>160:
            # if y[-1]>130:
            # if x[-1]<10:
            if y[-1]<10:
                plt.scatter(x, y,color='steelblue',s=10)

    plt.savefig(f'./stat_pic/xy_{name}_rounabout_c.png')
    plt.savefig(f'./stat_pic/xy_{name}_roundabout_c.svg')


def compare_stat(data_dir,scenario,fig_size=(20,10)):
    
    # plt.ylabel('y/m',fontsize=15)
    datas = load_data(data_dir)
    print(len(datas))
    i=0
    for data in tqdm(datas):
        speed,heading = [d[-2] for d in data],[d[-1] for d in data]
        x,y = [d[0] for d in data],[d[1] for d in data]
        t = np.arange(len(data))/10
        # if y[-1]>54 and y[-1]<56 :
        # if y[-1]<20 and x[-1]>160:
        # if y[-1]>130:
        if y[-1]<10:
            f,ax = plt.subplots(figsize=fig_size)
            plt.grid(True,linewidth=1.5)
            fontsize=15
            bwith = 2
            plt.tick_params(labelsize=20)
            plt.title('Speed',fontsize=30)
            
            
            plt.ylabel('speed(m/s)',fontsize=30)
            plt.xlabel('Time(s)',fontsize=30)
            ax.spines['bottom'].set_linewidth(bwith)
            ax.spines['left'].set_linewidth(bwith)
            ax.spines['top'].set_linewidth(bwith)
            ax.spines['right'].set_linewidth(bwith)
            ax.plot(t,speed,linewidth=3)
            ax.set_xlim(0,t[-1])
            ax.set_ylim(0,13)
            if not os.path.exists(f'./speed_res_pic/{scenario}'):
                os.makedirs(f'./speed_res_pic/{scenario}')
            plt.savefig(f'./speed_res_pic/{scenario}/speed_{i}.png')
            plt.savefig(f'./speed_res_pic/{scenario}/speed_{i}.svg')

            plt.close(fig=f)
            
            f,ax = plt.subplots(figsize=fig_size)
            
            plt.grid(True,linewidth=1.5)
            
            fontsize=15
            plt.tick_params(labelsize=20)

            ax.spines['bottom'].set_linewidth(bwith)
            ax.spines['left'].set_linewidth(bwith)
            ax.spines['top'].set_linewidth(bwith)
            ax.spines['right'].set_linewidth(bwith)
            plt.title('Heading',fontsize=30)
            plt.ylabel('Heading Angle(rad)',fontsize=30)
            plt.xlabel('Time(s)',fontsize=30)
            
            ax.plot(t,heading,linewidth=3)
            ax.set_xlim(0,t[-1])
            if not os.path.exists(f'./speed_res_pic/{scenario}'):
                os.makedirs(f'./speed_res_pic/{scenario}')
           
            plt.savefig(f'./speed_res_pic/{scenario}/heading_{i}.png')
            plt.savefig(f'./speed_res_pic/{scenario}/heading_{i}.svg')

            plt.close(fig=f)
        i+=1

def stat_data():
    data_list = glob('./test_results/*.json')
    
    for data_dir in data_list:
        with open(data_dir,'r',encoding='utf-8') as reader:
            datass = json.load(reader)
        datas = datass[-1]
        speed_list=[]
        for data in datas:
            speed = [d[-2] for d in data]
            speed_list.append(np.mean(speed))
        print(data_dir.split('/')[-1])
        print(f'mean:{np.mean(speed_list)},std:{np.std(speed_list)}')




