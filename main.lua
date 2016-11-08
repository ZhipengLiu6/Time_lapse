--learning temporal transformations from time-lapse videos, pairwise generation

require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'xlua'
require 'paths'
require 'torch'
debugger = require 'fb.debugger'
nngraph.setDebug(true)

opt = {
   dataset = 'time_lapse',       -- time_lapse
   batchSize = 100,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 100,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'AE',
   noise = 'normal',       -- uniform / normal
   kW = 5,	--The kernel width of the convolution
   kH = 5	,	--The kernel height of the convolution
   dW = 2, 	--The step of the convolution in the width dimension
   dH = 2,	--The step of the convolution in the height dimension
   nc = 3	,	--The channel of input image
   ncondition = 4, --The channel of condition term
   seed = 1000,     --- random seed
   datapath = '/home/zhipengliu/zhipengliu/ME/related work/time_lapse/imagenetTest.t7',  --data path
}

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
paths.dofile('data.lua')
dataset = getdata(opt.datapath, opt.batchSize)
local function weights_init(m)
   local name = torch.type(m)
   --debugger.enter()
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local function get_model(opt)
	local SpatialBatchNormalization = nn.SpatialBatchNormalization
	local SpatialConvolution = nn.SpatialConvolution
	local SpatialFullConvolution = nn.SpatialFullConvolution

	local nc = opt.nc
	local ndf = opt.ndf
	local ngf = opt.ngf
	local kW = opt.kW
	local kH = opt.kH
	local dW = opt.dW
	local dH = opt.dH
	local ncondition = opt.ncondition
	local encoder = nn.Sequential()
	-- input is nc x 64 x 64
	encoder:add(SpatialConvolution(nc, ndf, kW, kH, dW, dH))
	encoder:add(SpatialBatchNormalization(ndf))
	encoder:add(nn.ReLU(true))
	--state size: ndf x 30 x 30
	encoder:add(SpatialConvolution(ndf, ndf * 2, kW, kH, dW, dH))
	encoder:add(SpatialBatchNormalization(ndf * 2))
	encoder:add(nn.ReLU(true))
	--state size: (2 * ndf) x 13  x 13
	encoder:add(SpatialConvolution(ndf * 2, ndf * 4, kW, kH, dW, dH))
	encoder:add(SpatialBatchNormalization(ndf * 4))
	encoder:add(nn.ReLU(true))
	--state size: (4 * ndf) x 5 x 5
	encoder:add(SpatialConvolution(ndf * 4, ndf * 8, kW, kH, dW, dH))
	encoder:add(nn.Tanh())
	--state size: (8 * ndf) X 1 X 1

	encoder:add(nn.View(ndf * 8, 1, 1):setNumInputDims(3))
	--state size:(8 * ndf) x 1

	--input_image :RGB color image 3 * 64 *64
	local input_image = nn.Identity()() 
	--imgz: (8 * ndf) x 1 x 1
	local imgz = encoder(input_image)
	--input_condition: conditional term:one-hot code (4 * 1 * 1)
	local input_condition = nn.Identity()()
	--allz size:516 x 1 x1
	local allz = nn.JoinTable(2)({imgz, input_condition}) --the 1 dimension is batchsize
	--encoder input channel : nde = 8 * ndf + ncondition = 8 * 64 + 4 = 512 + 4 = 516
	local nde = ndf * 8 + ncondition

	local decoder_layer1 = nn.ReLU(true)(SpatialBatchNormalization(ndf * 4)(SpatialFullConvolution(nde, ndf * 4, kW, kH, dW, dH)(allz)))
	--output size: 256, (1-1) * 2 + kw = 5
	local decoder_layer2 = nn.ReLU(true)(SpatialBatchNormalization(ndf * 2)(SpatialFullConvolution(ndf * 4, ndf * 2, kW, kH, dW, dH)(decoder_layer1)))
	--output size: 128, (5 - 1) * 2 + 5 = 13
	local decoder_layer3 = nn.ReLU(true)(SpatialBatchNormalization(ndf)(SpatialFullConvolution(ndf * 2, ndf , kW, kH, dW, dH, 0, 0, 1, 1)(decoder_layer2)))
	--output size: 64, (13 - 1) * 2 + 5 + 1 = 30
	local decoder_image = nn.Tanh()(SpatialFullConvolution(ndf, nc, kW, kH, dW, dH, 0, 0, 1, 1)(decoder_layer3))
	--output size:3, (30 - 1) * 2 + 5 + 1 = 64

	AE = nn.gModule({input_image, input_condition}, {decoder_image})

	--graph.dot(AE.fg, 'AE forward', 'AE forward')
	return AE
end

print("buiding model ......")
local model = get_model(opt)
model.name = 'AE'
model:apply(weights_init)
print("building model Done")
local criterion = nn.MSECriterion()

optimconfig = {
   learningRate = opt.lr,
   beta1 = opt.beta1,	
}

------------------------------------------------------
local input_image = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local input_condition = torch.zeros(opt.batchSize, opt.ncondition, 1, 1)
--initialize the input_condition as 0
local start = 1
local i = 0
input_condition:apply(function()
	i = i + 1
	if (i - start) % opt.ncondition == 0 then
		return 1
	else
		return 0
	end
end)
assert(input_condition:sum() == opt.batchSize, 'input_condition initialize wrongly')

local output_image = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
------------------------------------------------------
if opt.gpu > 0 then
	print("Using GPU")
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	input_image = input_image:cuda(); input_condition:cuda();output_image = output_image:cuda()

	if pcall(require, 'cudnn') then
		require 'cudnn'
		print("cudnn success")
		cudnn.benchmark = true
		cudnn.convert(model, cudnn)
	end
	model:cuda(); criterion:cuda()
end
----------------------------------------------------

local parameters, gradParameters = model:getParameters()

local fx = function (x)
	collectgarbage()
	if x~= parameters then
		parameters:copy(x) -- get new parameters
	end
	--print("parameters:sum=", parameters:sum())
	gradParameters:zero() -- reset gradients
	data_tm:reset()
	data_tm:resume()
	local imagebatch = dataset:selectbatch()
	data_tm:stop()
	input_image:copy(imagebatch)
	output_image:copy(imagebatch)
	output_image:mul(2):add(-1)	--make output_image as (-1, 1)(tanh())

	local output = model:forward({input_image, input_condition})
	myerror = criterion:forward(output, output_image)
	local df_do = criterion:backward(output, output_image)
	--debugger.enter()
	model:backward({input_image, input_condition}, df_do)
	print("gradparameters:sum=", gradParameters:sum())
	return myerror, gradParameters
end

for epoch = 1, opt.niter do
	epoch_tm:reset()
	local counter = 0
	for i = 1, math.min(dataset:size(), opt.ntrain), opt.batchSize do --opt.train is useless now
		tm:reset()
		optim.adam(fx, parameters, optimconfig)
	       print(('Epoch: [%d] \t Time: %.3f  DataTime: %.3f  '
	                   .. '  Err: %.4f  '):format(epoch, tm:time().real, data_tm:time().real,myerror))
	       xlua.progress(i, math.min(dataset:size(), opt.ntrain))
	end
	if epoch > 0 then
		paths.mkdir('checkpoints')
		parameters, gradParameters = nil, nil --nil them to avoid spiking memory
		torch.save('checkpoints/'..opt.name..'_'..epoch..'net.t7', model:clearState())
		parameters, gradParameters = model:getParameters()
	end
   	print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
       	epoch, opt.niter, epoch_tm:time().real))	
   	xlua.progress(epoch, opt.niter)
end
os.execute('open -a  AE.svg')

