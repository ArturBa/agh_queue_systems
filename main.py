import math
import enum
from scipy.special import binom
from itertools import combinations  
import numpy as np


class QueueMMmFIFOInf:
    # M/M/m/FIFO/inf
    def __init__(self, _lambda, _mi, m):
        self._lambda = _lambda
        self._mi = _mi
        assert self.ro() < m
        self.m = m

    def ro(self):
        return float(self._lambda / self._mi)

    def p(self, number):
        if number > self.m:
            return self.ro()**(number) / (math.factorial(self.m) * self.m**(number - self.m) ) * self.p0()
        return self.ro() ** number / math.factorial(number) * self.p0()

    def p0(self):
        ro_k_sum = 0
        for k in range(self.m - 1):
            ro_k_sum += self.ro() ** k / math.factorial(k)
        return 1 / ( ro_k_sum + self.ro() ** self.m / (math.factorial(self.m - 1) * (self.m - self.ro())))

    def K(self):
        ro = self.ro()
        return ro + ro**(self.m+1) / ((self.m - ro)**2 *math.factorial(self.m - 1)) * self.p0()
    
    def T(self):
        return self.K() / self._mi



class QueueMMinf:
    # M/M/inf
    def __init__(self, _lambda, _mi):
        self._lambda = _lambda
        self._mi = _mi

    def ro(self):
        return float(self._lambda / self._mi)
        
    def p0(self):
        return math.exp(-1 * self.ro())

    def p(self, number):
        return self.p0() * self.ro()**number * 1/math.factorial(number)
    
    def K(self):
        return self.ro()

    def T(self):
        return 1/self._mi

class QueueMMmFIFOInfInd:
    # M/M/m/FIFO/Inf individual
    def __init__(self, _lambda, _mi):
        self._lambda = _lambda
        self._mi = _mi
        self.m = len(_mi)
        sum([self.ro(k) for k in range(self.m)]) < self.m

    def ro(self, k):
        assert 0 <= k <= self.m
        return float(self._lambda / self._mi[k])

    def p(self, k):
        assert k >= 0
        if k < self.m:
            return self.p0() * calcSK(k, self.m) / (math.factorial(k) * (binom(self.m, k))**(k-self.m))
        return self.p0() * calcSK(k, self.m) / (math.factorial(k) * calcSK(self.m -1, m)**(k-self.m))

    
    def p0(self):
        q = 0
        for i in range(self.m - 1):
            q += calcSK(i, self.m) / (math.factorial(i) * binom(self.m, i)) 
        return (1 + q + (calcSK(self.m, self.m) * calcSK(self.m-1, self.m)) / (math.factorial(math.m) * (calcSK(self.m-1, self.m) * calcSK(self.m, self.m))))

    def K(self):
        sk_1_m = calcSK(self.m-1, self.m)
        sk_m_m = calcSK(self.m, self.m)
        return self.p0() * sk_1_m / (math.factorial(self.m) * (sk_1_m/sk_m_m - 1)**2) + self.m * sk_m_m/sk_1_m

    def T(self):
        # How to calculate this stuff? Mean of the mi?
        self.K * self._mi[0]
        
def prod(val) :  
    res = 1 
    for ele in val:  
        res *= ele  
    return res   

def calcSK(k , ro):
    SK = 0
    comb = combinations(ro, k)
    SK += (prod(a) for a  in comb)
    return SK

class RestaurantSystem:
    def __init__(self):
        self.systems = [None] * (len(Systems) + 1)
        self.lambdas = [None] * len(Orders)
        self.p = [None] * len(Orders)
        self.systemTypes = [SystemTypes.Type1] * (len(Systems) + 1)

    def Waiter(self, _mi, m):
        l = self.lambdaIR(Systems.Waiter, Orders.Local) + self.lambdaIR(Systems.Waiter, Orders.Online)
        self.systems[Systems.Waiter] = QueueMMmFIFOInf(l, _mi, m)
    def Waiter2(self, _mi, m):
        l = self.lambdaIR(Systems.Waiter2, Orders.Local) + self.lambdaIR(Systems.Waiter2, Orders.Online)
        self.systems[Systems.Waiter2] = QueueMMmFIFOInf(l, _mi, m)
    def OnlineBuffer(self, _mi):
        l = self.lambdaIR(Systems.OnlineBuffer, Orders.Local) + self.lambdaIR(Systems.OnlineBuffer, Orders.Online)
        self.systemTypes[Systems.OnlineBuffer] = SystemTypes.Type3
        self.systems[Systems.OnlineBuffer] = QueueMMinf(l, _mi)
    def SoupChef(self, _mi):
        l = self.lambdaIR(Systems.ChefSoup, Orders.Local) + self.lambdaIR(Systems.ChefSoup, Orders.Online)
        self.systems[Systems.ChefSoup] = QueueMMmFIFOInfInd(l, _mi)
    def MainChef(self, _mi):
        l = self.lambdaIR(Systems.ChefMain, Orders.Local) + self.lambdaIR(Systems.ChefMain, Orders.Online)
        self.systems[Systems.ChefMain] = QueueMMmFIFOInfInd(l, _mi)
    def DesserChef(self, _mi):
        l = self.lambdaIR(Systems.ChefDesser, Orders.Local) + self.lambdaIR(Systems.ChefDesser, Orders.Online)
        self.systems[Systems.ChefDesser] = QueueMMmFIFOInfInd(l, _mi)
    def Barista(self, _mi):
        l = self.lambdaIR(Systems.Barista, Orders.Local) + self.lambdaIR(Systems.Barista, Orders.Online)
        self.systems[Systems.Barista] = QueueMMmFIFOInfInd(l, _mi)
    def Delivery(self, _mi, m):
        l = self.lambdaIR(Systems.Delivery, Orders.Local) + self.lambdaIR(Systems.Delivery, Orders.Online)
        self.systems[Systems.Delivery] = QueueMMmFIFOInf(l, _mi, m)
    def Cashier(self, _mi, m):
        l = self.lambdaIR(Systems.Cashier, Orders.Local) + self.lambdaIR(Systems.Cashier, Orders.Online)
        self.systems[Systems.Cashier] = QueueMMmFIFOInf(l, _mi, m)

    def Online(self, _lambda, p):
        self.lambdas[Orders.Online] = _lambda
        self.p[Orders.Online] = p
    def Local(self, _lambda):
        self.lambdas[Orders.Local] = _lambda
        self.p[Orders.Local] = p

    def P(self, r, p):
        self.p[r] = p

    def _lambda(self, r, l):
        self.lambdas[r] = l
    

    def Check(self):
        return  any(s is None for s in self.systems) or\
                any(p is None for p in self.p) or\
                any(l is None for l in self.lambdas)

    def lambdaR(self, order):
        """
        lambda for r-class
        """
        return sum(self.lambda0IR(i, order) for i in range(len(Systems)))

    def lambda0IR(self, system, order):
        return self.lambdas[order] * np.squeeze(np.asarray(self.p[order][0]))[system]

    def lambdaIR(self, system, order):
        suma = 0
        matrix = self.p[order].transpose()[system]
        matrix = np.squeeze(np.asarray(matrix))
        for n in range(len(matrix)):
            if (matrix[n] > 0):
                suma += matrix[n] * self.lambdaIR(n, order)
        return self.lambda0IR(system, order) + suma

    def P_I_Ki(self, system, ki):
        if(self.systemTypes[system] == SystemTypes.Type3):
            return self.P_I_Ki_Type_3(self, system, ki)
        if(self.systemTypes[system] == SystemTypes.Type1 and self.systems[system].m > 1):
            return self.P_I_Ki_Type_1(system, ki)
        return (1 - self.systems[system].ro(system)) * self.systems[system].ro(system) ** ki

    def P_I_Ki_Type_3(self, system, ki):
        return math.e**(- self.systems[system].ro(system)) * (self.systems[system].ro(system) ** ki)/(math.factorial(ki))

    def P_I_Ki_Type_1(self, system, ki):
        raise NotImplementedError

class Orders(enum.IntEnum):
    Online= 0
    Local = 1

class SystemTypes(enum.IntEnum):
    Type1 = 1
    Type2 = 2
    Type3 = 3
    Type4 = 4

class Systems(enum.IntEnum):
    Waiter = 1
    OnlineBuffer = 2
    ChefSoup = 3
    ChefMain = 4
    ChefDesser = 5
    Barista = 6
    Waiter2 = 7
    Delivery = 8
    Cashier = 9


"""
queMMn = QueueMMmFIFOInf(10, 13, 2)
print(f'średnia liczba zgłoszeń {queMMn.K()}')
print(f'średni czas oczekiwania {queMMn.T()}')

# średnia liczba zgłoszeń 0.9721528471528472
# średni czas oczekiwania 0.07478098824252671

queMMn = QueueMMinf(10, 13)
print(f'średnia liczba zgłoszeń {queMMn.K()}')
print(f'średni czas oczekiwania {queMMn.T()}')

# średnia liczba zgłoszeń 0.7692307692307693
# średni czas oczekiwania 0.07692307692307693
"""
