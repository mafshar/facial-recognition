local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model  = nn.Sequential()

model:add(Convolution(3, 16, 4, 4))
model:add(ReLU())
model:add(Convolution(16, 32, 4, 4))
model:add(ReLU())
model:add(Max(2,2,2,2))

model:add(Convolution(32, 64, 4, 4))
model:add(ReLU())
model:add(Convolution(64, 128, 4, 4))
model:add(ReLU())
model:add(Max(2,2,2,2))

model:add(nn.Reshape(21632))
model:add(Linear(21632, 4096))
model:add(Linear(4096, 321))
return model
