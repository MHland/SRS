"""
@name: Xinanjiang model
@author: Wei Haoshan
@Time: 2022/11/19
@dependencies: numpy
"""
import time
from test_indicators import NSE
import numpy as np
from SRS_python import SRS
from numba import jit
import warnings
from base_code import warm_up_code, warm_up_type1, transform_to_vector, \
                                to_float_numpy, to_float_number
from flow_routing_routine import MuskingumMethod, JitIndex

class XAJ_model:
    # OPEN JIT OR NOT
    __jit_index = JitIndex.jit_index
    def __init__(self):
        # ASSIGN INPUT VARIABLES
        self.__Prec, self.__ETp = 2.0, 1.0
        # ASSIGN STATE VARIABLES
        self.__WU, self.__WL, self.__WD, self.__FR = 2.0, 5.0, 10.0, 0.3
        self.__S, self.__TRSS, self.__TRG, self.__fip, self.__fop = 10.0, 5.0, 5.0, 5.0, 5.0
        # ASSIGN PARAMETERS
        self.__SM, self.__KG, self.__KSS = 35.0, 0.15, 0.5
        self.__KKG, self.__KKSS, self.__WUM = 0.96, 0.5, 50.0
        self.__WLM, self.__WDM, self.__IMP = 150.0, 50.0, 0.05
        self.__B, self.__C, self.__EX = 0.15, 1.2, 0.25
        self.__XE, self.__KE, self.__KC = 0.25, 48.0, 0.5
        # VECTORIZATION OR NOT
        self.__vectorize_index = False
        self.first_time = True

    @property
    def vectorization(self):
        return self.__vectorize_index

    @vectorization.setter
    def vectorization(self, new_value):
        if isinstance(new_value, bool):
            self.__vectorize_index = new_value
        else:
            warnings.warn('vectorization is a BOOL value! ')

    def set_input(self, Prec=2.0, ETp=1.0):
        '''
        Set the inputs of the XAJ model
        :param Prec: Precipitation (mm), type: list, numpy, float, or int
        :param ETp: Evaporation (mm), type: list, numpy, float, or int
        :return: None
        '''
        if self.__vectorize_index:
            self.__Prec, self.__ETp = to_float_numpy(Prec, ETp)
        else:
            self.__Prec, self.__ETp = to_float_number(Prec, ETp)

    def set_state(self, WU=2.0, WL=5.0, WD=10.0, FR=0.3, S=10.0, TRSS=5.0, TRG=5.0, fip=5.0, fop=5.0):
        '''
        Pass the model state value or set the initial value of the model state value.
        :param WU:
        :param WL:
        :param WD:
        :param FR:
        :param S:
        :param TRSS:
        :param TRG:
        :param fip:
        :param fop:
        :return: None
        '''
        if self.first_time:
            if self.__vectorize_index:
                self.__WU, self.__WL, self.__WD, self.__FR, self.__S, self.__TRSS, self.__TRG, self.__fip, self.__fop = to_float_numpy(
                    WU, WL, WD, FR, S, TRSS, TRG, fip, fop)
            else:
                self.__WU, self.__WL, self.__WD, self.__FR,self.__S, self.__TRSS, self.__TRG, self.__fip, self.__fop = to_float_number(
                    WU, WL, WD, FR, S, TRSS, TRG, fip, fop)
        else:
            self.__WU, self.__WL, self.__WD, self.__FR, self.__S, self.__TRSS, self.__TRG, self.__fip, self.__fop = \
                WU, WL, WD, FR, S, TRSS, TRG, fip, fop
    def set_parameter(self, SM=35.0, KG=0.15, KSS=0.5, KKG=0.96, KKSS=0.5,
                      WUM=50.0, WLM=150.0, WDM=50.0, IMP=0.05, B=0.5, C=0.15,
                      EX=1.2, XE=0.25, KE=48.0, KC=0.5):
        '''
        Set the parameter values of the model.
        :param SM:
        :param KG:
        :param KSS:
        :param KKG:
        :param KKSS:
        :param WUM:
        :param WLM:
        :param WDM:
        :param IMP:
        :param B:
        :param C:
        :param EX:
        :param XE:
        :param KE:
        :param KC:
        :return: None
        '''
        if self.first_time:
            if self.__vectorize_index:
                self.__SM, self.__KG, self.__KSS = to_float_numpy(SM, KG, KSS)
                self.__KKG, self.__KKSS, self.__WUM = to_float_numpy(KKG, KKSS, WUM)
                self.__WLM, self.__WDM, self.__IMP = to_float_numpy(WLM, WDM, IMP)
                self.__B, self.__C, self.__EX = to_float_numpy(B,C,EX)
                self.__XE, self.__KE, self.__KC = to_float_numpy(XE,KE,KC)
            else:
                self.__SM, self.__KG, self.__KSS = to_float_number(SM, KG, KSS)
                self.__KKG, self.__KKSS, self.__WUM = to_float_number(KKG, KKSS, WUM)
                self.__WLM, self.__WDM, self.__IMP = to_float_number(WLM, WDM, IMP)
                self.__B, self.__C, self.__EX = to_float_number(B, C, EX)
                self.__XE, self.__KE, self.__KC = to_float_number(XE, KE, KC)
        else:
            self.__SM, self.__KG, self.__KSS = SM, KG, KSS
            self.__KKG, self.__KKSS, self.__WUM = KKG, KKSS, WUM
            self.__WLM, self.__WDM, self.__IMP = WLM, WDM, IMP
            self.__B, self.__C, self.__EX = B, C, EX
            self.__XE, self.__KE, self.__KC = XE, KE, KC

    @staticmethod
    @jit(nopython=__jit_index)
    def __calculate_others_method(Prec, ETp, KC, WUM, WLM, WDM, IMP, WU, WL, WD, B):
        ETp = ETp * KC
        WM = WUM + WLM + WDM
        WMM = (1 + B) * WM / (1 - IMP)
        W = WU + WL + WD
        PE = Prec - ETp
        # It may be bug, if PE can be divisible by 5：
        PE[PE / 5 == np.floor(PE / 5)] = PE[PE / 5 == np.floor(PE / 5)] + 0.01
        index = PE.shape
        return ETp, WM, WMM, W, PE, index

    @staticmethod
    @jit(nopython=__jit_index)
    def __calculate_others_scalar_method(Prec, ETp, KC, WUM, WLM, WDM, IMP, WU, WL, WD, B):
        ETp = ETp * KC
        WM = WUM + WLM + WDM
        WMM = (1 + B) * WM / (1 - IMP)
        W = WU + WL + WD
        PE = Prec - ETp
        # It may be bug, if PE can be divisible by 5：
        if abs(PE/5) == int(abs(PE/5)):
            PE = PE+0.01
        return ETp, WM, WMM, W, PE

    def __calculate_others(self):
        # CALCULATE OTHER VARIABLES
        if self.__vectorize_index:
            self.__ETp, self.__WM, self.__WMM, self.__W, self.__PE, self.__index = \
                self.__calculate_others_method(self.__Prec, self.__ETp, self.__KC, self.__WUM, self.__WLM, self.__WDM,
                                               self.__IMP, self.__WU, self.__WL, self.__WD, self.__B)
        else:
            self.__ETp, self.__WM, self.__WMM, self.__W, self.__PE = \
                self.__calculate_others_scalar_method(self.__Prec, self.__ETp, self.__KC, self.__WUM, self.__WLM, self.__WDM,
                                               self.__IMP, self.__WU, self.__WL, self.__WD, self.__B)

    @staticmethod
    @jit(nopython=__jit_index)
    def __calculation_on_step_T_method(index, PE, Prec, ETp, WU, WL, WD, WM, WLM, WUM, WMM, W, B, C):
        G = np.zeros((index))
        PE_leq5 = PE <= 5
        PE_ge5 = PE > 5
        G[PE_leq5] = 1.0
        G[PE_ge5] = np.floor(PE[PE_ge5] / 5) + 1
        PED = np.zeros((index[0], int(np.max(G))))
        PED[PE_leq5, 0] = PE[PE_leq5]
        for i in range(index[0]):
            if G[i] > 1:
                PED[i, :int(G[i] - 1)] = 5.0
                PED[i, int(G[i])] = PE[i] - (G[i] - 1) * 5.0
        # Runoff generation calculation in 1-day step and 1/(PE(T)/5) day time step
        rd = np.zeros((PED.shape))
        ## + Calculation of actual ET
        EU, ED, EL = np.zeros((index)), np.zeros((index)), np.zeros((index))
        # for i in range(index[0]):
        #     PED[i,:], G[i], rd[i,:], EU[i], ED[i], EL[i], WU[i], WL[i], WD[i] = process1(rd[i,:], G[i], PED[i,:], PE[i], Prec[i], ETp[i], WU[i], WL[i], WD[i], WM[i], WLM[i], WUM[i], WMM[i], W[i], B[i], C[i])
        for i in range(index[0]):
            if PE[i] <= 0:
                rd[i, :G[i]] = 0.0
                # + Calculation of actual ET part1 ：
                if WU[i] + PE[i] > 0:
                    EU[i] = ETp[i]
                    ED[i] = 0
                    EL[i] = 0
                    WU[i] = WU[i] + PE[i]
                else:
                    EU[i] = WU[i] + Prec[i]
                    WU[i] = 0
                    if WL[i] > C[i] * WLM[i]:
                        EL[i] = (ETp[i] - EU[i]) * WL[i] / WLM[i]
                        WL[i] = WL[i] - EL[i]
                        ED[i] = 0
                    else:
                        if WL[i] > C[i] * (ETp[i] - EU[i]):
                            EL[i] = C[i] * (ETp[i] - EU[i])
                            WL[i] = WL[i] - EL[i]
                            ED[i] = 0
                        else:
                            EL[i] = WL[i]
                            WL[i] = 0
                            if C[i] * (ETp[i] - EU[i]) - EL[i] > WD[i]:
                                ED[i] = WD[i]
                            else:
                                ED[i] = C[i] * (ETp[i] - EU[i]) - EL[i]
                            WD[i] = WD[i] - ED[i]
            else:
                if W[i] >= WM[i]:
                    A = WMM[i]
                else:
                    A = WMM[i] * (1 - np.power((1 - W[i] / WM[i]), (1 / (1 + B[i]))))
                R = 0
                peds = 0
                for j in range(int(G[i])):
                    A = A + PED[i, j]
                    peds = peds + PED[i, j]
                    rii = R
                    R = peds - WM[i] + W[i]
                    if A < WMM[i]:
                        R = R + WM[i] * ((1 - A / WMM[i]) ** (1 + B[i]))
                    rd[i, j] = R - rii
                # + Calculation of actual ET part2：
                EU[i] = ETp[i]
                ED[i] = 0
                EL[i] = 0
                if WU[i] + PE[i] - R < WUM[i]:
                    WU[i] = WU[i] + PE[i] - R
                else:
                    if WU[i] + WL[i] + PE[i] - WUM[i] > WLM[i]:
                        WU[i] = WUM[i]
                        WL[i] = WLM[i]
                        WD[i] = W[i] + PE[i] - R - WU[i] - WL[i]
                    else:
                        WU1 = WU[i]
                        WU[i] = WUM[i]
                        WL[i] = WU1 + WL[i] + PE[i] - R - WUM[i]
        return PED, G, rd, EU, ED, EL, WU, WL, WD

    @staticmethod
    @jit(nopython=__jit_index)
    def __calculation_on_step_T_scalar_method(PE, Prec, ETp, WU, WL, WD, WM, WLM, WUM, WMM, W, B, C):
        if PE <= 5:
            G = 1.0
            PED = [PE]*int(G)
        else:
            G = int(PE/5)+1
            PED = [5.0]*G
            PED[-1] = PE-(G-1)*5
        rd = [0.0]*int(G)
        if PE <= 0:
            R = 0.0
        else:
            if W >= WM:
                A = WMM
            else:
                A = WMM * (1.0-(1.0-W/WM)**(1.0/(1.0+B)))
            R = 0.0
            peds = 0.0
            for k in range(int(G)):
                A = A+PED[k]
                peds = peds+PED[k]
                rri = R
                R = peds-WM+W
                if A < WMM:
                    R = R+WM*((1.0-A/WMM)**(1.0+B))
                rd[k] = R-rri
        if PE < 0:
            if WU+PE > 0:
                EU = ETp
                ED = 0.0
                EL = 0.0
                WU = WU+PE
            else:
                EU = WU+Prec
                WU = 0.0
                if WL > C*WLM:
                    EL = (ETp-EU)*WL/WLM
                    WL = WL-EL
                    ED = 0.0
                else:
                    if WL>C*(ETp-EU):
                        EL = C*(ETp-EU)
                        WL = WL-EL
                        ED = 0.0
                    else:
                        EL = WL
                        WL = 0.0
                        s1 = C*(ETp-EU)-EL
                        if s1 < WD:
                            ED = s1
                        else:
                            ED = WD
                        WD = WD-ED
        else:
            EU = ETp
            ED = 0.0
            EL = 0.0
            if WU+PE-R<WUM:
                WU = WU+PE-R
            else:
                if WU+WL+PE-WUM>WLM:
                    WU = WUM
                    WL = WLM
                    WD = W+PE-R-WU-WL
                else:
                    WU_1 = WU
                    WU = WUM
                    WL = WU_1+WL+PE-R-WUM
        return np.array(PED), G, np.array(rd), EU, ED, EL, WU, WL, WD

    def __calculation_on_step_T(self):
        '''
        Calculation depending on the time step, T.
        '''
        if self.__vectorize_index:
            self.__PED, self.__G, self.__rd, self.__EU, self.__ED, self.__EL, self.__WU, \
            self.__WL, self.__WD = self.__calculation_on_step_T_method(self.__index, self.__PE, self.__Prec, self.__ETp,
                                                                       self.__WU, self.__WL, self.__WD, self.__WM,
                                                                       self.__WLM, self.__WUM, self.__WMM, self.__W,
                                                                       self.__B, self.__C)
        else:
            self.__PED, self.__G, self.__rd, self.__EU, self.__ED, self.__EL, self.__WU, \
            self.__WL, self.__WD = self.__calculation_on_step_T_scalar_method(self.__PE, self.__Prec, self.__ETp,
                                                                       self.__WU, self.__WL, self.__WD, self.__WM,
                                                                       self.__WLM, self.__WUM, self.__WMM, self.__W,
                                                                       self.__B, self.__C)
        ## Calculation of actual ET
        self.__AET = self.__EU + self.__EL + self.__ED
        self.__W = self.__WU + self.__WL + self.__WD

    @staticmethod
    @jit(nopython=__jit_index)
    def __separation_of_runoff_components_method(index, PE, IMP, PED, FR, EX, S, SM, rd, G, KG, KSS, TRSS, TRG, KKG, KKSS):
        SMM = (1 + EX) * SM
        PE_leq0 = PE <= 0
        PE_ge0 = PE > 0
        RS, RG, RSS, rb, AU = np.zeros((index)), np.zeros((index)), np.zeros((index)), \
                              np.zeros((index)), np.zeros((index))
        RS[PE_leq0] = 0
        RG[PE_leq0] = S[PE_leq0] * KG[PE_leq0] * FR[PE_leq0]
        RSS[PE_leq0] = RG[PE_leq0] * KSS[PE_leq0] / KG[PE_leq0]
        S[PE_leq0] = S[PE_leq0] * (1 - KG[PE_leq0] - KSS[PE_leq0])
        rb[PE_ge0] = IMP[PE_ge0] * PE[PE_ge0]
        KSSD, KGD = np.zeros((index)), np.zeros((index))
        KSSD[PE_ge0] = (1 - np.power(1 - (KG[PE_ge0] + KSS[PE_ge0]), 1 / G[PE_ge0])) / (1 + KG[PE_ge0] / KSS[PE_ge0])
        KGD[PE_ge0] = KSSD[PE_ge0] * KG[PE_ge0] / KSS[PE_ge0]
        for i in range(index[0]):
            if PE[i] > 0:
                for j in range(int(G[i])):
                    td = rd[i, j] - IMP[i] * PED[i, j]
                    X = FR[i]
                    FR[i] = td / PED[i, j]
                    S[i] = X * S[i] / FR[i]
                    if S[i] >= SM[i]:
                        AU[i] = SMM[i]
                    else:
                        AU[i] = SMM[i] * (1 - (1 - S[i] / SM[i]) ** (1 / (1 + EX[i])))
                    if AU[i] + PED[i, j] < SMM[i]:
                        RR = (PED[i, j] - SM[i] + S[i] + SM[i] * ((1 - (PED[i, j] + AU[i]) / SMM[i]) ** (1 + EX[i]))) * \
                             FR[i]
                    else:
                        RR = (PED[i, j] + S[i] - SM[i]) * FR[i]
                    RS[i] = RR + RS[i]
                    S[i] = PED[i, j] - RR / FR[i] + S[i]
                    RG[i] = S[i] * KGD[i] * FR[i] + RG[i]
                    RSS[i] = S[i] * KSSD[i] * FR[i] + RSS[i]
                    S[i] = S[i] * (1 - KGD[i] - KSSD[i])
        RS[PE_ge0] = RS[PE_ge0] + rb[PE_ge0]

        TRS = RS
        TRSS = TRSS * KKSS + RSS * (1 - KKSS)
        TRG = TRG * KKG + RG * (1 - KKG)
        Qsim = TRS + TRSS + TRG
        return FR, S, TRS, TRSS, TRG, Qsim

    @staticmethod
    @jit(nopython=__jit_index)
    def __separation_of_runoff_components_scalar_method(PE, IMP, PED, FR, EX, S, SM, rd, G, KG, KSS, TRSS, TRG, KKG, KKSS):
        SMM = (1+EX)*SM
        if PE<=0:
            RS = 0.0
            RG = S*KG*FR
            RSS = RG*KSS/KG
            S = S*(1.0-KG-KSS)
        else:
            rb = IMP*PE
            KSSD = (1.0-(1.0-(KG+KSS))**(1.0/G))/(1.0+KG/KSS)
            KGD = KSSD*KG/KSS
            RS = 0.0
            RG = 0.0
            RSS = 0.0
            for j in range(int(G)):
                td = rd[j]-IMP*PED[j]
                X = FR
                FR = td/PED[j]
                S = X*S/FR
                if S >= SM:
                    AU = SMM
                else:
                    AU = SMM*(1.0-(1.0-S/SM)**(1.0/(1.0+EX)))

                if AU+PED[j] < SMM:
                    RR = (PED[j]-SM+S+SM*(1.0-(PED[j]+AU)/SMM)**(1.0+EX))*FR
                else:
                    RR = (PED[j]+S-SM)*FR
                RS = RR+RS
                S = PED[j]-RR/FR+S
                RG = S*KGD*FR+RG
                RSS = S*KSSD*FR+RSS
                S = S*(1.0-KGD-KSSD)
            RS = RS+rb

        TRS = RS
        TRSS = TRSS * KKSS + RSS * (1.0 - KKSS)
        TRG = TRG * KKG + RG * (1.0 - KKG)
        Qsim = TRS + TRSS + TRG
        return FR, S, TRS, TRSS, TRG, Qsim

    def __separation_of_runoff_components(self):
        if self.__vectorize_index:
            self.__FR, self.__S, self.__TRS, self.__TRSS, self.__TRG, self.__Qsim \
                = self.__separation_of_runoff_components_method(self.__index, self.__PE, self.__IMP, self.__PED, self.__FR,
                                                                self.__EX, self.__S, self.__SM, self.__rd, self.__G,
                                                                self.__KG, self.__KSS, self.__TRSS, self.__TRG, self.__KKG,
                                                                self.__KKSS)
        else:
            self.__FR, self.__S, self.__TRS, self.__TRSS, self.__TRG, self.__Qsim \
                = self.__separation_of_runoff_components_scalar_method(self.__PE, self.__IMP, self.__PED,
                                                                self.__FR,
                                                                self.__EX, self.__S, self.__SM, self.__rd, self.__G,
                                                                self.__KG, self.__KSS, self.__TRSS, self.__TRG,
                                                                self.__KKG,
                                                                self.__KKSS)

    def __route_simulates(self):
        ''' Route simulates daily flow (rtot) to catchment outlet (rtot)
        '''
        if self.__vectorize_index:
            self.__Qsim_out, self.__fip, self.__fop = MuskingumMethod.route_simulate_vector_method(self.__index, self.__Qsim,
                                                                                    self.__KE, self.__XE, self.__fip,
                                                                                    self.__fop)
        else:
            self.__Qsim_out, self.__fip, self.__fop = MuskingumMethod.route_simulates_scalar_method(self.__Qsim,
                                                                                    self.__KE, self.__XE, self.__fip,
                                                                                    self.__fop)

    def XAJ(self):
        '''
        Xinanjiang model process
        :return:
        '''
        self.__calculate_others()
        self.__calculation_on_step_T()
        self.__separation_of_runoff_components()
        self.__route_simulates()
        state = {'WU': self.__WU, 'WL': self.__WL, 'WD': self.__WD, 'FR': self.__FR, 'S': self.__S, 'TRSS': self.__TRSS,
                 'TRG': self.__TRG, 'fip': self.__fip, 'fop': self.__fop}
        out = {'AET': self.__AET, 'TRS': self.__TRS, 'TRSS': self.__TRSS, 'TRG': self.__TRG, 'Qsim': self.__Qsim,
               'Qsim_out': self.__Qsim_out}
        return state, out

    def run(self, Prec, ETp, initial_state, parameter):
        '''
        Using the Xinanjiang Model
        :param Prec:
        :param ETp:
        :param initial_state:
        :param parameter:
        :return:
        '''
        self.first_time = True
        self.set_parameter(SM=parameter['SM'], KG=parameter['KG'], KSS=parameter['KSS'],
                           KKG=parameter['KKG'], KKSS=parameter['KKSS'], WUM=parameter['WUM'],
                           WLM=parameter['WLM'], WDM=parameter['WDM'], IMP=parameter['IMP'],
                           B=parameter['B'], C=parameter['C'], EX=parameter['EX'], XE=parameter['XE'],
                           KE=parameter['KE'], KC=parameter['KC'])
        self.set_state(WU=initial_state['WU'], WL=initial_state['WL'], WD=initial_state['WD'],
                       FR=initial_state['FR'], S=initial_state['S'], TRSS=initial_state['TRSS'],
                       TRG=initial_state['TRG'], fip=initial_state['fip'], fop=initial_state['fop'])
        s = np.array(parameter['SM']).shape
        if type(Prec) != list:
            Prec = Prec.tolist()
            ETp = ETp.tolist()
        l = len(Prec)
        if s == ():
            s = 1
        else:
            s = max(np.array(parameter['SM']).shape)
        AET, TRS, TRSS, TRG, Qsim, Qsim_out = np.zeros((l,s)),np.zeros((l,s)),np.zeros((l,s)),np.zeros((l,s)),np.zeros((l,s)),np.zeros((l,s))
        for i in range(l):
            self.set_input(Prec=Prec[i], ETp=ETp[i])
            state, out = self.XAJ()
            self.first_time = False
            self.set_state(WU=state['WU'], WL=state['WL'], WD=state['WD'],
                           FR=state['FR'], S=state['S'], TRSS=state['TRSS'],
                           TRG=state['TRG'], fip=state['fip'], fop=state['fop'])
            AET[i,:], TRS[i,:], TRSS[i,:], TRG[i,:], Qsim[i,:], Qsim_out[i,:] = out['AET'], out['TRS'], out['TRSS'], out['TRG'], out['Qsim'], out['Qsim_out']
        out = {'AET': AET, 'TRS': TRS, 'TRSS': TRSS, 'TRG': TRG, 'Qsim': Qsim,
               'Qsim_out': Qsim_out}
        return out

    def target_function(self, x, Prec, ETp, Q_obs, initial_state, Vectorization=True):
        '''
        Objective function for model calibration
        :param x:
        :param Prec:
        :param ETp:
        :param Q_obs:
        :param initial_state:
        :return:
        '''
        if Vectorization:
            x = np.array(x)
            s = x.shape[0]
            parameter = {'SM': x[:,0], 'KG': x[:,1], 'KSS': x[:,2], 'KKG': x[:,3], 'KKSS': x[:,4],
                     'WUM': x[:,5], 'WLM': x[:,6], 'WDM': x[:,7], 'IMP': x[:,8], 'B': x[:,9], 'C': x[:,10],
                     'EX': x[:,11], 'XE': x[:,12], 'KE': x[:,13], 'KC': x[:,14]}
            state, Prec, ETp = transform_to_vector(s, initial_state, Prec, ETp)
            self.vectorization = Vectorization
            out = self.run(Prec=Prec, ETp=ETp, initial_state=state, parameter=parameter)
            Q_sim = out['Qsim_out']
            y = np.zeros((s,1))
            y[:,0] = 1-NSE(Q_sim=Q_sim, Q_obs=Q_obs)
        else:
            try:
                parameter = {'SM': x[0,0], 'KG': x[0,1], 'KSS': x[0,2], 'KKG': x[0,3], 'KKSS': x[0,4],
                             'WUM': x[0,5], 'WLM': x[0,6], 'WDM': x[0,7], 'IMP': x[0,8], 'B': x[0,9], 'C': x[0,10],
                             'EX': x[0,11], 'XE': x[0,12], 'KE': x[0,13], 'KC': x[0,14]}
            except:
                parameter = {'SM': x[0], 'KG': x[1], 'KSS': x[2], 'KKG': x[3], 'KKSS': x[4],
                             'WUM': x[5], 'WLM': x[6], 'WDM': x[7], 'IMP': x[8], 'B': x[9],
                             'C': x[10], 'EX': x[11], 'XE': x[12], 'KE': x[13], 'KC': x[14]}
            state, Prec, ETp = transform_to_vector(1, initial_state, Prec, ETp)
            self.vectorization = False
            out = self.run(Prec=Prec, ETp=ETp, initial_state=state, parameter=parameter)
            Q_sim = out['Qsim_out']
            y = 1 - NSE(Q_sim=Q_sim, Q_obs=Q_obs)
        return y

    def calibrate(self, Prec=None, ETp=None, Q_obs=None,  initial_state=None,
                      bl=None, bu=None, p=5, sp=3, delta=0.3,deps=5, num=None,
                      Vectorization=True, ShowProgress=True, type_result=True,
                      warm_up=False, warm_up_year=2):
        '''

        :param Prec:
        :param ETp:
        :param Q_obs:
        :param initial_state:
        :param bl:
        :param bu:
        :param p:
        :param sp:
        :param delta:
        :param deps:
        :param num:
        :param Vectorization:
        :param ShowProgress:
        :return:
        '''
        n = 15
        if Prec is None or ETp is None or Q_obs is None:
            raise TypeError('Prec, ETp or Q_obs is None, they should be list or numpy!')
        if bl is None:
            bl = np.array([10, 0.1, 0.3, 0.95, 0.1, 5, 25, 5, 0, 0, 0.1, 1.0, 0, 24, 0])
            bu = np.array([60, 0.2, 0.7, 0.99, 0.9, 100, 300, 100, 0.1, 1, 0.2, 1.5, 0.5, 72, 1])
        else:
            bl = np.array(bl)
            bu = np.array(bu)
        bl[bl == 0] = 0.0001
        if num is None:
            if Vectorization:
                num = 300
            else:
                num = 10000
        if initial_state is None:
            initial_state = {'WU': 0, 'WL': 10.0, 'WD': 20.0, 'FR': 0.1, 'S': 0.01, 'TRSS': 0.05,
                             'TRG': 0.02, 'fip': 0.02, 'fop': 0.02}
        Prec, ETp, Q_obs, Q_obs_pre, warm_up_time, l = warm_up_code(Prec, ETp, Q_obs, warm_up, warm_up_year)
        srs = SRS(n, bl, bu, p=p, sp=sp, delta=delta, deps=deps, MAX=False, Vectorization=Vectorization, num=num, ShowProgress=ShowProgress)
        srs.run(self.target_function, Prec, ETp, Q_obs, initial_state, Vectorization)
        x = srs.get_result()[1][1]
        parameter = {'SM': x[0], 'KG': x[1], 'KSS': x[2], 'KKG': x[3], 'KKSS': x[4],
                     'WUM': x[5], 'WLM': x[6], 'WDM': x[7], 'IMP': x[8], 'B': x[9], 'C': x[10],
                     'EX': x[11], 'XE': x[12], 'KE': x[13], 'KC': x[14]}
        if type_result:
            warm_up_type1(srs, parameter)
        return parameter, initial_state

if __name__=='__main__':
    # print(help(XAJ_model))
    parameter = {'SM':32.5, 'KG':0.15, 'KSS':0.5, 'KKG':0.97, 'KKSS':0.5,
                      'WUM':52.5, 'WLM':175.0, 'WDM':52.2, 'IMP':0.05, 'B':0.5, 'C':0.15,
                      'EX':1.25, 'XE':0.25, 'KE':48.0, 'KC':0.5}
    Prec = [2 for i in range(1000)]
    ETp = [4 for i in range(1000)]
    initial_state = {'WU':0, 'WL':10.0, 'WD':20.0, 'FR':0.1, 'S':0.01, 'TRSS':0.05,
                     'TRG':0.02, 'fip':0.02, 'fop':0.02}
    X1 = XAJ_model()
    X1.vectorization = False
    t1 = time.time()
    # Get runoff (may 15s)
    for i in range(1000):
        out = X1.run(Prec=Prec,ETp=ETp,initial_state=initial_state,parameter=parameter)
    print(time.time()-t1)
    # Calibration (vectorization) (may 25s)
    t1 = time.time()
    X1.calibrate(Prec=Prec, ETp=ETp, initial_state=initial_state, Q_obs=out['Qsim_out'], num=100, Vectorization=True)
    print(time.time() - t1)