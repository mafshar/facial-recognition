local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model  = nn.Sequential()

-- local x = torch.Tensor(3, 100, 100)

model:add(Convolution(3, 100, 7, 7))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Dropout(0.2))

model:add(Convolution(100, 150, 4, 4))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Dropout(0.2))

model:add(Convolution(150, 250, 4, 4))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Dropout(0.2))

-- print(#model:forward(x))
-- os.exit()

model:add(nn.Reshape(20250))
model:add(Linear(20250, 300))
model:add(Linear(300, 50))
return model
