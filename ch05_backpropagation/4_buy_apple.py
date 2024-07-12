from ch05_backpropagation.layers import MulLayer

apple = 100		# 사과 1개 가격
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파 -> 이때 초기화 됨
apple_price = mul_apple_layer.forward(apple, apple_num)
total_price = mul_tax_layer.forward(apple_price, tax)
print(total_price)	# 220

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# dL/d사과가격, dL/d사과개수, dL/d세금비율
print(dapple, dapple_num, dtax)	# 2.2 110, 200