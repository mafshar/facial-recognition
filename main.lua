require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'

local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 100, 'batchsize')
cmd:option('-epochs', 10 , 'epochs')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
local dbg   = require 'debugger'

---------------------- PREPROCESSING ----------------------
-- TODO: defining the data read and setting up iterators
local base_data_path = "/Users/mohammadafshar1/Desktop/Fall 2016/Computer_Vision/Project/facial-recognition/"

datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}
trainiterator = getCifarIterator(datasets)
datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_5.t7', 'ascii')}
validiterator = getCifarIterator(datasets)
datasets = {torch.load(base_data_path .. 'cifar-10-torch/test_batch.t7', 'ascii')}
testiterator  = getCifarIterator(datasets)

local function getCifarIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
        local list = torch.range(1, dataset.data:size(1)):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
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
			  input  = x.input:double():reshape(3,32,32),
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

---------------------- NETWORK ARCHITECTURE ----------------------

-- TODO: defining correct backprop methodology based on
-- slide 32 of FACE REC
local network = require("./model/model.lua")
local criterion = nn.CrossEntropyCriterion()
local lr = config.lr
local epochs = config.epochs

print("Started training!")

for epoch = 1, epochs do
    -- if epoch == 2 then break end
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

    local validloss = 0
    local validerrors = 0
    count = 0
    for d in validiterator() do
        network:forward(d.input)
        criterion:forward(network.output, d.target)

        validloss = validloss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        validerrors = validerrors + (pred:size(1) - pred:eq(d.target):sum())
    end
    validloss = validloss / count

    print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f - valid | validloss: %2.4f | validerror: %2.4f | s/iter: %2.4f',
    epoch, lr, loss, errors, validloss, validerrors, timer:time().real
    ))

    if epoch > 0 then
        theta = (network:get(1)).weight
        theta = image.toDisplayTensor(w)
        theta = image.scale(w, 15*w:size(3), 15*w:size(2), 'simple')
        image.saveJPG('epoch_' .. epoch .. '_filters.jpg', w)
    end
    -- for mnist
    -- if epoch == 10 then
    --     local weights = torch.reshape(network.weight, 10, 28, 28)
    --     out_image_weights = image.toDisplayTensor(weights)
    --     image.saveJPG('./network_weights.jpg', out_image_weights)
    -- end

end

local testerrors = 0
for d in testiterator() do
    network:forward(d.input)
    criterion:forward(network.output, d.target)
    local _, pred = network.output:max(2)
    testerrors = testerrors + (pred:size(1) - pred:eq(d.target):sum())
end
