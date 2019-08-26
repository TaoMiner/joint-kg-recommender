import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def drawDBbook():
    width = .55 # width of a bar
    '''
    m1_t = pd.DataFrame({
    'users' : [4770, 812, 276, 123, 59],
    'cfkg' : [13.17, 11.59, 9.31, 7.67, 5.54],
    'cofm_s' : [13.5, 13, 9.89, 7.86, 5.62],
    'cofm_r' : [13.38, 12.81, 9.95, 7.81, 5.64]})
    '''
    m1_t = pd.DataFrame({
    'users' : [8, 12, 16, 17],
    'fm' : [3.92, 5.96, 9.88, 10.6],
    'bprmf' : [4.15, 6.27, 9.82, 10.7],
    'cfkg' : [3.53, 5.59, 8.42, 8],
    'cke' : [3.82, 7.31, 12.23, 13.8],
    'cofm_s' : [3.51, 6.51, 9.89, 10.13],
    'cofm_r' : [3.62, 5.95, 9.58, 11.09],
    'transup_h' : [3.80, 5.99, 9.70, 10.42],
    'transup_s' : [4.32, 6.45, 10.19, 11.90],
    'jtransup_h' : [4.24, 7.21, 12.24, 13.74],
    'jtransup_s' : [4.22, 7.49, 12.07, 13.78],
    })

    m1_t['fm'].plot(secondary_y=True, color = '#95e1d3', linestyle='-', marker='o')
    m1_t['bprmf'].plot(secondary_y=True, color = '#8b4c8c', linestyle='-', marker='^')
    m1_t['cfkg'].plot(secondary_y=True, color = '#0d627a', linestyle='-', marker='s')
    m1_t['cke'].plot(secondary_y=True, color = '#0c907d', linestyle='-', marker='s')
    m1_t['cofm_s'].plot(secondary_y=True, color = '#cce490', linestyle='-', marker='s')
    m1_t['cofm_r'].plot(secondary_y=True, color = '#f2f4b2', linestyle='-', marker='s')
    m1_t['transup_h'].plot(secondary_y=True, color = '#f57665', linestyle='-', marker='s')
    m1_t['transup_s'].plot(secondary_y=True, color = '#48466d', linestyle='-', marker='s')
    m1_t['jtransup_h'].plot(secondary_y=True, color = '#3d84a8', linestyle='-', marker='s')
    m1_t['jtransup_s'].plot(secondary_y=True, color = '#46cdcf', linestyle='-', marker='s')
    ax = m1_t['users'].plot(kind='bar', width = width, color = '#abedd8')

    plt.xlim([-width, len(m1_t['users'])-width])
    ax.set_xticklabels(('2858', '1370', '877', '111'), rotation = 45)
    plt.legend()
    plt.show()

def drawMl1m():
    width = .55 # width of a bar
    
    m1_t = pd.DataFrame({
    'Ratings' : [17, 30, 50, 70, 89, 123, 174, 244, 347, 563],
    'FM' : [7.55, 10.9, 14.11, 15.6, 15.88, 14.78, 13.4, 11.64, 9.86, 7.66],
    'BPRMF' : [8.33, 12.4, 15.29, 16.59, 16.81, 15.42, 13.92, 11.98, 10, 7.67],
    'CFKG' : [7.62, 12.3, 15.7, 15.35, 15.16, 14.23, 13.04, 11.36, 9.52, 7.68],
    'CKE' : [10.76, 15.92, 19.32, 20.07, 19.63, 18.66, 16.71, 14.49, 11.67, 8.27],
    # 'CoFM_s' : [7.59, 11.96, 15.31, 16.09, 16.3, 15.88, 14.57, 12.67, 10.32, 7.88],
    'CoFM' : [7.46, 11.63, 15.36, 16.14, 16.22, 15.67, 14.64, 12.44, 10.4, 7.91],
    'TUP' : [11.66, 17.04, 20.26, 19.46, 19.56, 18.07, 15.61, 13.33, 10.9, 8.01],
    # 'TransUP_s' : [11.16, 16.94, 20.7, 19.48, 19.12, 18.17, 15.54, 13.27, 10.75, 7.92],
    # 'jTransUP_h' : [10.14, 15.81, 18.63, 19.78, 20, 18.87, 17.76, 15.22, 12.06, 8.34],
    'KTUP' : [10.08, 15.78, 19.21, 20, 20.17, 19.65, 17.94, 15.18, 11.77, 8.34]
    })

    ax1 = m1_t['FM'].plot(secondary_y=True, color = '#8293ff', linestyle='--', marker='^')
    m1_t['BPRMF'].plot(secondary_y=True, color = '#503bff', linestyle='--', marker='p')

    m1_t['CFKG'].plot(secondary_y=True, color = '#f08a5d', linestyle='-', marker='*')
    m1_t['CKE'].plot(secondary_y=True, color = '#b83b5e', linestyle='-', marker='x')
    # m1_t['cofm_s'].plot(secondary_y=True, color = '#cce490', linestyle='-', marker='s')
    m1_t['CoFM'].plot(secondary_y=True, color = '#6a2c70', linestyle='-', marker='+')
    m1_t['TUP'].plot(secondary_y=True, color = '#48466d', linestyle='--', marker='o')
    # m1_t['TransUP_s'].plot(secondary_y=True, color = '#48466d', linestyle='--', marker='o')

    # m1_t['jtransup_h'].plot(secondary_y=True, color = '#393e46', linestyle='-', marker='s')
    m1_t['KTUP'].plot(secondary_y=True, color = '#222831', linestyle='-', marker='s')
    ax = m1_t['Ratings'].plot(kind='bar', width = width, color = '#abedd8')
    
    ax.set_xlabel('User Group')
    ax.set_ylabel('# Avg. Ratings', color='#45b7b7')
    ax1.set_ylabel('F1 score (%)')

    plt.xlim([-width, len(m1_t['Ratings'])-width])
    # ax.set_xticklabels(('684', '1364', '773', '585', '438', '750', '410', '517', '234', '285'), rotation = 45)
    plt.legend()
    plt.show()

def drawMl1mTopnPrec():
    width = .55 # width of a bar
    
    prec = pd.DataFrame({
    'Topn' : [1, 2, 5, 10, 20, 50, 100],
    'FM' : [33.54, 31.68, 28.1, 24.54, 20.92, 15.89, 12.22],
    'BPRMF' : [34.92, 32.88, 29.26, 25.74, 21.97, 16.64, 12.74],
    'CFKG' : [32.5, 30.6, 27.52, 24.56, 21.16, 16.24, 12.39],
    'CKE' : [39.11, 36.92, 34.16, 30.91, 26.60, 20.11, 14.8],
    'CoFM' : [29.09, 29.13, 28.18, 26.12, 22.95, 17.57, 13.24],
    'TransUP' : [38.53, 36.95, 33.64, 30.19, 26.13, 19.66, 14.54],
    'jTransUP' : [40.84, 39.38, 35.55,31.62, 26.97, 19.95, 14.45]
    })

    ax1 = prec['FM'].plot(secondary_y=False, color = '#8293ff', linestyle='--', marker='^')
    prec['BPRMF'].plot(secondary_y=False, color = '#503bff', linestyle='--', marker='p')

    prec['CFKG'].plot(secondary_y=False, color = '#f08a5d', linestyle='-', marker='*')
    prec['CKE'].plot(secondary_y=False, color = '#b83b5e', linestyle='-', marker='x')
    # m1_t['cofm_s'].plot(secondary_y=True, color = '#cce490', linestyle='-', marker='s')
    prec['CoFM'].plot(secondary_y=False, color = '#6a2c70', linestyle='-', marker='+')
    # m1_t['transup_h'].plot(secondary_y=True, color = '#f57665', linestyle='-', marker='s')
    prec['TransUP'].plot(secondary_y=False, color = '#48466d', linestyle='--', marker='o')

    # m1_t['jtransup_h'].plot(secondary_y=True, color = '#393e46', linestyle='-', marker='s')
    prec['jTransUP'].plot(secondary_y=False, color = '#222831', linestyle='-', marker='s')
    
    ax1.set_xlabel('Topn')
    ax1.set_ylabel('Precision@N')
    ax1.set_xticks([0,1,2,3,4,5,6])
    ax1.set_xticklabels(('1', '2', '5', '10', '20', '50', '100'))

    # plt.legend()
    plt.show()

def drawMl1mTopnRecall():
    width = .55 # width of a bar
    recall = pd.DataFrame({
    'Topn' : [1, 2, 5, 10, 20, 50, 100],
    'FM' : [1.72, 3.16, 6.63, 10.84, 17.48, 30.53, 44.32],
    'BPRMF' : [1.83, 3.35, 7.15, 11.77, 18.81, 32.56, 46.7],
    'CFKG' : [1.73, 3.14, 6.64, 11.29, 18.38, 32.19, 45.65],
    'CKE' : [2.2, 3.94, 8.54, 14.64, 23.2, 39.41, 53.25],
    'CoFM' : [1.35, 2.7, 6.44, 11.54, 19.27, 34.04, 47.94],
    'TransUP' : [2.29, 4.25, 9.05, 15.08, 24.09, 40.46, 54.92],
    'jTransUP' : [2.25, 4.18, 8.81, 14.61, 23.01, 38.19, 51.38]
    })
    ax1 = recall['FM'].plot(secondary_y=False, color = '#8293ff', linestyle='--', marker='^')
    recall['BPRMF'].plot(secondary_y=False, color = '#503bff', linestyle='--', marker='p')

    recall['CFKG'].plot(secondary_y=False, color = '#f08a5d', linestyle='-', marker='*')
    recall['CKE'].plot(secondary_y=False, color = '#b83b5e', linestyle='-', marker='x')
    # m1_t['cofm_s'].plot(secondary_y=True, color = '#cce490', linestyle='-', marker='s')
    recall['CoFM'].plot(secondary_y=False, color = '#6a2c70', linestyle='-', marker='+')
    # m1_t['transup_h'].plot(secondary_y=True, color = '#f57665', linestyle='-', marker='s')
    recall['TransUP'].plot(secondary_y=False, color = '#48466d', linestyle='--', marker='o')

    # m1_t['jtransup_h'].plot(secondary_y=True, color = '#393e46', linestyle='-', marker='s')
    recall['jTransUP'].plot(secondary_y=False, color = '#222831', linestyle='-', marker='s')
    
    ax1.set_xlabel('Topn')
    ax1.set_ylabel('Recall@N')
    ax1.set_xticks([0,1,2,3,4,5,6])
    ax1.set_xticklabels(('1', '2', '5', '10', '20', '50', '100'))

    # plt.legend()
    plt.show()

def drawMl1mTopnF1():
    width = .55 # width of a bar
    f1 = pd.DataFrame({
    'Topn' : [1, 2, 5, 10, 20, 50, 100],
    'FM' : [3.1, 5.24, 9.17, 12.27, 15.12, 16.85, 16.07],
    'BPRMF' : [3.29, 5.55, 9.78, 13.16, 16.11, 17.76, 16.8],
    'CFKG' : [3.11, 5.18, 9.11, 12.59, 15.57, 17.37, 16.34],
    'CKE' : [3.94, 6.49, 11.66, 16.19, 19.74, 21.48, 19.39],
    'CoFM' : [2.44, 4.51, 8.97, 13.08, 16.68, 18.72, 17.43],
    'TransUP' : [4.08, 6.92, 12.09, 16.32, 19.94, 21.41, 19.34],
    'jTransUP' : [4.04, 6.89, 12.07, 16.34, 19.8, 21.1, 18.83]
    })
    ax1 = f1['FM'].plot(secondary_y=False, color = '#8293ff', linestyle='--', marker='^')
    f1['BPRMF'].plot(secondary_y=False, color = '#503bff', linestyle='--', marker='p')

    f1['CFKG'].plot(secondary_y=False, color = '#f08a5d', linestyle='-', marker='*')
    f1['CKE'].plot(secondary_y=False, color = '#b83b5e', linestyle='-', marker='x')
    # m1_t['cofm_s'].plot(secondary_y=True, color = '#cce490', linestyle='-', marker='s')
    f1['CoFM'].plot(secondary_y=False, color = '#6a2c70', linestyle='-', marker='+')
    # m1_t['transup_h'].plot(secondary_y=True, color = '#f57665', linestyle='-', marker='s')
    f1['TransUP'].plot(secondary_y=False, color = '#48466d', linestyle='--', marker='o')

    # m1_t['jtransup_h'].plot(secondary_y=True, color = '#393e46', linestyle='-', marker='s')
    f1['jTransUP'].plot(secondary_y=False, color = '#222831', linestyle='-', marker='s')
    
    ax1.set_xlabel('Topn')
    ax1.set_ylabel('F1@N')
    ax1.set_xticks([0,1,2,3,4,5,6])
    ax1.set_xticklabels(('1', '2', '5', '10', '20', '50', '100'))

    plt.legend()
    plt.show()

def drawMl1mTopnHits():
    width = .55 # width of a bar
    hits = pd.DataFrame({
    'Topn' : [1, 2, 5, 10, 20, 50, 100],
    'CFKG' : [3.25, 11.23, 23.33, 32.08, 41.13, 53.18, 62.38],
    'CKE' : [1.66, 6.9, 17.15, 25.01, 33.28, 45.18, 54.78],
    'CoFM' : [3.32, 12.67, 26.77, 36.46, 45.67, 56.91, 64.98],
    'transe' : [3.28, 13.03, 27.23, 36.59, 45.48, 56.33, 64.07],
    'transh' : [3.43, 13.67, 27.97, 37.25, 45.92, 56.62, 64.35],
    'transr' : [3.27, 10.48, 21.22, 29.14, 37.49, 48.94, 57.87],
    'jtransup' : [3.38, 13.64, 28.28, 38.05, 47.07, 57.78, 65.14]
    })
    ax1 = hits['transe'].plot(secondary_y=False, color = '#8293ff', linestyle='--', marker='^')
    hits['transh'].plot(secondary_y=False, color = '#503bff', linestyle='--', marker='p')

    hits['transr'].plot(secondary_y=False, color = '#f08a5d', linestyle='-', marker='*')
    hits['CFKG'].plot(secondary_y=False, color = '#f08a5d', linestyle='-', marker='*')
    hits['CKE'].plot(secondary_y=False, color = '#b83b5e', linestyle='-', marker='x')
    # m1_t['cofm_s'].plot(secondary_y=True, color = '#cce490', linestyle='-', marker='s')
    hits['CoFM'].plot(secondary_y=False, color = '#6a2c70', linestyle='-', marker='+')
    # m1_t['transup_h'].plot(secondary_y=True, color = '#f57665', linestyle='-', marker='s')

    # m1_t['jtransup_h'].plot(secondary_y=True, color = '#393e46', linestyle='-', marker='s')
    hits['jtransup'].plot(secondary_y=False, color = '#222831', linestyle='-', marker='s')
    
    ax1.set_xlabel('Topn')
    ax1.set_ylabel('F1@N')
    ax1.set_xticks([0,1,2,3,4,5,6])
    ax1.set_xticklabels(('1', '2', '5', '10', '20', '50', '100'))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    drawMl1m()
    # drawDBbook()
    # drawMl1mTopnPrec()
    # drawMl1mTopnRecall()
    # drawMl1mTopnF1()
    # drawMl1mTopnHits()