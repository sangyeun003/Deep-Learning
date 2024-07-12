from ch05_backpropagation.layers import MulLayer, AddLayer

apple = 100		# 사과 1개 가격
apple_num = 2
orange = 150	# 귤 1개 가격
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(all_price, tax)

# 역전파
dprice = 1		# dL/dL
dall_price, dtax = mul_tax_layer.backward(dprice)	#dL/dall_price, dL/dtax
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)	# dL/dapple_price, dL/dorange_price
dorange, dorange_num = mul_orange_layer.backward(dorange_price)		# dL/dorange, dL/dorange_num
dapple, dapple_num = mul_apple_layer.backward(dapple_price)		# dL/dapple, dL/dapple_num

print(total_price)		# 715
print(dapple_num, dapple, dorange, dorange_num, dtax)	# 110 2.2 3.3 165 650