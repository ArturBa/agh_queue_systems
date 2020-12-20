from restaurant import *
from probability import p_local, p_online


rest = RestaurantSystem()

rest.P(Orders.Local, np.matrix(p_local))
rest.P(Orders.Online, np.matrix(p_online))
rest._lambda(Orders.Local, 200)
rest._lambda(Orders.Online, 200)

# Restaurant setup
rest.Waiter(100, 3)
rest.OnlineBuffer(300)
rest.SoupChef([250, 210])
rest.MainChef([200, 210])
rest.DesserChef([200, 210])
rest.Barista([300])
rest.Waiter2(210, 2)
rest.Delivery(55, 4)
rest.Cashier(120, 2)



print(rest.lambda0IR(Systems.Waiter, Orders.Local))
print(rest.lambda0IR(Systems.Barista, Orders.Local))
print(rest.lambdaR(Orders.Local))
print(rest.lambdaIR(Systems.ChefDesser, Orders.Local))
print(rest.K_IR(Systems.OnlineBuffer, Orders.Online))
print(rest.K_IR(Systems.Barista, Orders.Local))