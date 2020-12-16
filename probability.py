# Online 
p_online = [
#System, Enter    Waiter1, OnlineBuffer, ChefSoup, ChefMain, ChefDessert, Barista, Delivery, Waiter2, Cashier,   Exit
        [ 0,        0,          1,          0,          0,          0,        0,      0 ,	0,	0,	0], # Enter	
        [ 0,        0,          0,          0,          0,          0,        0,      0 ,       0,	0,	0], # Waiter1	
        [ 0,        0,          0,         0.3,        0.5,        0.2,       0,      0 ,	0,	0,	0], # Online Buffer
        [ 0,        0,          0,          0,          0,          0,        0,      1 ,	0,	0,	0], # Chef Soup
        [ 0,        0,          0,          0,          0,          0,        0,      1 ,	0,	0,	0], # Chef Main
        [ 0,        0,          0,          0,          0,          0,        0,      1 ,	0,	0,	0], # Chef Dessert
        [ 0,        0,          0,          0,          0,          0,        0,      0 ,	0,	0,	0], # Barista        
        [ 0,        0,          0,          0,          0,          0,        0,      0 ,	0,	0,	1], # Delivery
        [ 0,        0,          0,          0,          0,          0,        0,      0 ,       0,	0,	0], # Waiter2
        [ 0,        0,          0,          0,          0,          0,        0,      0 ,	0,	0,	0], # Cashier   
]
# Local 
p_local = [
#System, Enter    Waiter1, OnlineBuffer, ChefSoup, ChefMain, ChefDessert, Barista, Delivery, Waiter2, Cashier,  Exit
	[ 0,        1,         0,          0,          0,          0,        0,      0 ,        0,	0,	0], # Enter
        [ 0,        0,         0,         0.3,        0.4,        0.1,      0.2,     0 ,        0,	0,      0], # Waiter1
        [ 0,        0,         0,          0,          0,          0,        0,      0 ,	0,	0,	0], # Online Buffer       
        [ 0,        0,         0,          0,         0.4,        0.1,      0.1,     0 ,	0.4,    0,     	0], # Chef Soup
        [ 0,        0,         0,          0,          0,         0.4,      0.1,     0 ,	0.5,    0,	0], # Chef Main
        [ 0,        0,         0,          0,          0,          0,       0.5,     0 ,	0.5,    0,	0], # Chef Dessert
        [ 0,        0,         0,          0,          0,          0,        0,      0 ,	1,	0,	0], # Barista
        [ 0,        0,         0,          0,          0,          0,        0,      0 ,	0,	0,	0], # Delivery
        [ 0,        0,         0,          0,          0,          0,        0,      0 ,        0,	1,	0], # Waiter2        
        [ 0,        0,         0,          0,          0,          0,        0,      0 ,	0,	0,	1], # Cashier
]
    