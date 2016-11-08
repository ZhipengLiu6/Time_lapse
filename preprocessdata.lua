--resize the imagenet data to 64 x 64
require 'torch'
require 'image'
require 'paths'
require 'nn'
require 'nngraph'

lapp = require 'pl.lapp'
local args = lapp [[
   preprocess the image net: resized to 64 x 64
     --path  (default "string")  imagenet data path
   ]]

function resizeImageNet(path)
	for imagename in paths.files(path, function(name) return name:find('.JPEG') end)  do
		local imagepath = paths.concat(path, imagename)
		print(imagepath)
		local im = image.load(imagepath)
		--image.display(im)
		local width = 64
		local height = 64
		local scaled = image.scale(im, width, height)
		image.save(imagepath, scaled)
	end
end

function TableToTensor(table)
  local tensorSize = table[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  merge=nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(table)
end

function saveTot7( path )
	local data = {}
	for imagename in paths.files(path, function(name) return name:find('.JPEG') end) do
		local imagepath = paths.concat(path, imagename)
		print(imagepath)
		local im = image.load(imagepath)
		table.insert(data, im)
	end
	savefile = '/media/zhipengliu/zhipeng/dataset/imagenetTest.t7'
	torch.save(savefile, TableToTensor(data))
end



