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
local base_data_path = "/Users/mohammadafshar1/Desktop/Fall 2016/Computer_Vision/Project/facial-recognition/data/"
-- local base_data_path = "/Users/mattoor/Desktop/Dropbox/School/Senior/Fall\ 2016/Computer\ Vision/facial-recognition/"


datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
		torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}

-- local all_images = torch.Tensor(750, 3, 100, 100)
-- local all_labels = torch.Tensor(750)

-- ndx = 1
-- for folder = 1, 50 do
--     folder_name = folder
--     if (folder < 10 ) then
--         folder_name = '0'.. folder
--     end
--
--     for image_ndx = 1, 15 do
--         img_name = image_ndx .. '.jpg'
--         if (image_ndx < 10) then
--             img_name = '0' .. image_ndx .. '.jpg'
--         end
--         img = image.load(base_data_path .. 'training/' .. folder_name .. '/' .. img_name )
-- 		-- image.display(img)
--         all_images[ndx] = img
-- 		all_labels[ndx] = folder
--         ndx = ndx + 1
--     end
-- end

-- FLATTENED VERSION OF ABOVE CODE

local all_images = torch.Tensor(750, 30000)
local all_labels = torch.Tensor(750)

local ndx = 1
for folder = 1, 50 do
    folder_name = folder
    if (folder < 10 ) then
        folder_name = '0'.. folder
    end

    for image_ndx = 1, 15 do
        img_name = image_ndx .. '.jpg'
        if (image_ndx < 10) then
            img_name = '0' .. image_ndx .. '.jpg'
        end
        img = image.load(base_data_path .. 'training/' .. folder_name .. '/' .. img_name )
		-- image.display(img)
        all_images[ndx] = img:view(img:nElement())
		all_labels[ndx] = folder
        ndx = ndx + 1
    end
end



local labels_shuffle = torch.randperm((#all_labels)[1])

-- create train set:
local train_data = {
   data = torch.Tensor(750, 3, 100, 100),
   labels = torch.Tensor(750),
   size = function() return 750 end
}

for i=1, 750 do
   train_data.data[i] = all_images[labels_shuffle[i]]:clone()
   train_data.labels[i] = all_labels[labels_shuffle[i]]
end

-- train_display.data   = train_mnist.data[{{1, 100}, {}, {}, {}}]
-- train_display.labels = train_mnist.labels[{{1, 100}}]

local datasets = {[1]={}, [2]={}, [3]={}}
datasets[1].data = train_data.data[{{1, 250}, {}, {}, {}}]
datasets[2].data = train_data.data[{{251, 500}, {}, {}, {}}]
datasets[3].data = train_data.data[{{501, 750}, {}, {}, {}}]

datasets[1].labels = train_data.labels[{{1, 250}}]
datasets[2].labels = train_data.labels[{{251, 500}}]
datasets[3].labels = train_data.labels[{{501, 750}}]

-- print(dataset)
-- os.exit()

-- datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_5.t7', 'ascii')}
-- validiterator = getCifarIterator(datasets)
-- datasets = {torch.load(base_data_path .. 'cifar-10-torch/test_batch.t7', 'ascii')}
-- testiterator  = getCifarIterator(datasets)



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
			  input  = x.input:double():reshape(3,100,100),
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
-- for sample in trainiterator:run() do
-- 	print(sample)
-- end

---------------------- NETWORK ARCHITECTURE ----------------------

-- TODO: defining correct backprop methodology based on
-- slide 32 of FACE REC
local network = require("./model/model.lua")
local criterion = nn.CrossEntropyCriterion()
local lr = 0.1
local epochs = 1

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

    -- local validloss = 0
    -- local validerrors = 0
    -- count = 0
    -- for d in validiterator() do
    --     network:forward(d.input)
    --     criterion:forward(network.output, d.target)
	--
    --     validloss = validloss + criterion.output --criterion already averages over minibatch
    --     count = count + 1
    --     local _, pred = network.output:max(2)
    --     validerrors = validerrors + (pred:size(1) - pred:eq(d.target):sum())
    -- end
    -- validloss = validloss / count

    -- print(string.format(
    -- 'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f - valid | validloss: %2.4f | validerror: %2.4f | s/iter: %2.4f',
    -- epoch, lr, loss, errors, validloss, validerrors, timer:time().real
    -- ))

	print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f | s/iter: %2.4f',
    epoch, lr, loss, errors, timer:time().real
    ))


    -- for mnist
    -- if epoch == 10 then
    --     local weights = torch.reshape(network.weight, 10, 28, 28)
    --     out_image_weights = image.toDisplayTensor(weights)
    --     image.saveJPG('./network_weights.jpg', out_image_weights)
    -- end

end
--
-- local testerrors = 0
-- for d in testiterator() do
--     network:forward(d.input)
--     criterion:forward(network.output, d.target)
--     local _, pred = network.output:max(2)
--     testerrors = testerrors + (pred:size(1) - pred:eq(d.target):sum())
-- end
