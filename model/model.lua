local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model  = nn.Sequential()

local x = torch.Tensor(3, 70, 70)

model:add(Convolution(3, 16, 4, 4))
-- model:add(ReLU())
model:add(Tanh())
model:add(Max(2,2,2,2))

model:add(Convolution(16, 32, 4, 4))
-- model:add(ReLU())
model:add(Tanh())
-- model:add(Max(3,3,2,2))

model:add(Convolution(32, 128, 7, 7))
model:add(ReLU())
model:add(Max(2,2,2,2))
-- model:add(Dropout(0.2))

-- print(#model:forward(x))
-- os.exit()


model:add(nn.Reshape(18432))
model:add(Linear(18432, 4096))
model:add(Linear(4096, 321))
return model
