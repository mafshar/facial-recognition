require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'
require 'io'

local cmd = torch.CmdLine()
cmd:option('-model', 1, 'model to test')
cmd:option('-edition', 1, 'model configuration')
-- cmd:option('-epochs', 10 , 'epochs')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
local dbg   = require 'debugger'

local base_data_path = "/Users/mohammadafshar1/Desktop/Fall 2016/Computer_Vision/Project/facial-recognition/data/"

-- local network = torch.load('./model/history/' .. config.model .. '/config' .. config.edition)
local network = torch.load('./model/history/1/cnn_model_afshar_mohammad')

local mlp = nn.CosineDistance()
local threshold = 0.7

local file = io.open(base_data_path .. "pairsDevTest.txt")
local num_misclassified = 0
local count = 1
if file then
    for line in file:lines() do
        if (count <= 501) and (count > 1) then --right
            local name, img1, img2 = unpack(line:split("\t"))
            img1 = tonumber(img1)
            img2 = tonumber(img2)
            -- print(img1)
            -- print(img2)
            if img1 < 10 then
                img1_filepath = base_data_path .. 'lfw/' .. name .. '/' .. name .. '_000' .. img1 .. '.jpg'
            else
                img1_filepath = base_data_path .. 'lfw/' .. name .. '/' .. name .. '_00' .. img1 .. '.jpg'
            end
            if img2 < 10 then
                img2_filepath = base_data_path .. 'lfw/' .. name .. '/' .. name .. '_000' .. img2 .. '.jpg'
            else
                img2_filepath = base_data_path .. 'lfw/' .. name .. '/' .. name .. '_00' .. img2 .. '.jpg'
            end
            image1 = image.load(img1_filepath)
            image2 = image.load(img2_filepath)
            local feat_vec1 = network:forward(image1).modules[-2].output -- second to last layer
            local feat_vec2 = network:forward(image2).modules[-2].output -- second to last layer
            print(feat_vec1)
            print(feat_vec2)
            prediction = mlp:forward({feat_vec1, feat_vec2})
            print(prediction)
            if prediction < threshold then
                num_misclassified = num_misclassified + 1
            end
        else --wrong
            local name1, img1, name2,  img2 = unpack(line:split("\t"))
            img1 = tonumber(img1)
            img2 = tonumber(img2)
            if img1 < 10 then
                img1_filepath = base_data_path .. 'lfw/' .. name1 .. '/' .. name1 .. '_000' .. img1 .. '.jpg'
            else
                img1_filepath = base_data_path .. 'lfw/' .. name1 .. '/' .. name1 .. '_00' .. img1 .. '.jpg'
            end
            if img2 < 10 then
                img2_filepath = base_data_path .. 'lfw/' .. name2 .. '/' .. name2 .. '_000' .. img2 .. '.jpg'
            else
                img2_filepath = base_data_path .. 'lfw/' .. name .. '/' .. name2 .. '_00' .. img2 .. '.jpg'
            end
            image1 = image.load(img1_filepath)
            image2 = image.load(img2_filepath)
            local feat_vec1 = network:forward(image1).modules[-2].output -- second to last layer
            local feat_vec2 = network:forward(image2).modules[-2].output -- second to last layer
            print(feat_vec1)
            print(feat_vec2)
            prediction = mlp:forward({feat_vec1, feat_vec2})
            print(prediction)
            if prediction > threshold then
                num_misclassified = num_misclassified + 1
            end
        end
        count = count + 1
    end
end
