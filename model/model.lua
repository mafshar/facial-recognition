local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3, 128, 7, 7))
model:add(Tanh())
model:add(Max(2,2,2,2))

model:add(Convolution(128, 256, 4, 4))
model:add(Tanh())
model:add(Max(2,2,2,2))

model:add(Convolution(256, 512, 4, 4))
model:add(Tanh())
model:add(Max(2,2,2,2))

model:add(nn.Reshape(512*3*3))
model:add(Linear(4608, 300))
model:add(Linear(300, 43))

return model
