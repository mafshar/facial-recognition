require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'

local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 256, 'batchsize')
cmd:option('-epochs', 10 , 'epochs')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
local dbg   = require 'debugger'

---------------------- PREPROCESSING ----------------------
local base_data_path = "/Users/mohammadafshar1/Desktop/Fall 2016/Computer_Vision/Project/facial-recognition/data/"
-- local base_data_path = "/Users/mattoor/Desktop/Dropbox/School/Senior/Fall\ 2016/Computer\ Vision/facial-recognition/"

-- datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
-- 		torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
-- 		torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
-- 		torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}

local all_images = torch.Tensor(41656, 14700)
local all_labels = torch.Tensor(1, 41656)

local network = require("./model/model.lua")
local criterion = nn.CrossEntropyCriterion()
local lr = config.lr
local epochs = config.epochs


-- actors
local ndx = 1
for folder = 1, 213 do
	dirname = './data/training-actor/' .. folder
	f = io.popen('ls ' .. dirname)
    for file in f:lines() do
		img_filename = dirname .. '/' .. file
        img = image.load(img_filename)
        all_images[ndx] = img:view(img:nElement())
		all_labels[{{}, ndx}] = folder - 1
        ndx = ndx + 1
    end
end

-- actress
for folder = 1, 108 do
	dirname = './data/training-actress/' .. folder
	f = io.popen('ls ' .. dirname)
    for file in f:lines() do
		img_filename = dirname .. '/' .. file
        img = image.load(img_filename)
		-- print(img_filename)
		-- print(#img)
        all_images[ndx] = img:view(img:nElement())
		all_labels[{{}, ndx}] = folder + 212
        ndx = ndx + 1
    end
end

local labels_shuffle = torch.randperm(41656)

-- create train set:
local train_data = {

   data = torch.Tensor(41656, 14700),
   labels = torch.Tensor(1, 41656),
   size = function() return 41656 end
}

for i=1, 41656 do
   train_data.data[i] = all_images[labels_shuffle[i]]:clone()
   train_data.labels[{{}, i}] = all_labels[{ {1}, {labels_shuffle[i]} }]
end

local datasets = {[1]={}, [2]={}, [3]={}, [4]={}}

datasets[1].data = torch.reshape(train_data.data[{{1, 10414}}], 14700, 10414):byte()
datasets[2].data = torch.reshape(train_data.data[{{10415, 20828}}], 14700, 10414):byte()
datasets[3].data = torch.reshape(train_data.data[{{20829, 31242}}], 14700, 10414):byte()
datasets[4].data = torch.reshape(train_data.data[{{31243, 41656}}], 14700, 10414):byte()

datasets[1].labels = train_data.labels[{ {1}, {1, 10414} }]:byte()
datasets[2].labels = train_data.labels[{ {1}, {10415, 20828} }]:byte()
datasets[3].labels = train_data.labels[{ {1}, {20829, 31242} }]:byte()
datasets[4].labels = train_data.labels[{ {1}, {31243, 41656} }]:byte()


local function getCifarIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
		-- print(_) --index
		-- print(dataset) --actual data/labels
        local list = torch.range(1, 10414):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
							-- print(idx)
							-- print(dataset.data[{{}, 251}])
                            return {
                                input  = dataset.data[{{}, idx}],
                                target = dataset.labels[{{}, idx}]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
		       return {
			  input  = x.input:double():reshape(3,70,70),
            --   input  = x.input:double(),
			  target = x.target:long():add(1),
		       }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end


trainiterator = getCifarIterator(datasets)

---------------------- NETWORK ARCHITECTURE ----------------------

local network = require("./model/model.lua")
local criterion = nn.CrossEntropyCriterion()
local lr = config.lr
local epochs = config.epochs

print("Started training!")

for epoch = 1, epochs do
    local timer = torch.Timer()
    local loss = 0
    local errors = 0
    local count = 0
    for d in trainiterator() do
        network:forward(d.input)
        criterion:forward(network.output, d.target)
        network:zeroGradParameters()
        criterion:backward(network.output, d.target)
        network:backward(d.input, criterion.gradInput)
        network:updateParameters(lr)
        loss = loss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        errors = errors + (pred:size(1) - pred:eq(d.target):sum())
    end
    loss = loss / count
	print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f | s/iter: %2.4f',
    epoch, lr, loss, errors, timer:time().real
    ))
end
torch.save("./model/history/2/config1", network)
