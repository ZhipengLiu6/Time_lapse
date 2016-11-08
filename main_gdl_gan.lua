--learning temporal transformations from time-lapse videos, pairwise generation

require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'xlua'
require 'paths'
require 'torch'
require 'GDLCriterion.lua'
debugger = require 'fb.debugger'
nngraph.setDebug(true)

opt = {
   dataset = 'time_lapse',       -- time_lapse
   batchSize = 100,
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
   gdlweight = 1,	--the weight of gdl loss function 
   pmseweight = 1, -- the weight of pixel-wise mean square loss fuction
   gdlalpha = 2, --1 means abs ,2 means square error in gdl loss function
   advweight = 0.01, -- the weight of adv loss function in generator model
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

local function get_G_model(opt)
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
local  function get_D_model(opt)
	local SpatialBatchNormalization = nn.SpatialBatchNormalization
	local SpatialConvolution = nn.SpatialConvolution

	local nc = opt.nc
	local ndf = opt.ndf
	local ngf = opt.ngf
	local kW = opt.kW
	local kH = opt.kH
	local dW = opt.dW
	local dH = opt.dH
	local ncondition = opt.ncondition
	local netD = nn.Sequential()
	-- input is (nc) x 64 x 64
	netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 32 x 32
	netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*2) x 16 x 16
	netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*4) x 8 x 8
	netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*8) x 4 x 4
	netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
	netD:add(nn.Sigmoid())
	-- state size: 1 x 1 x 1
	netD:add(nn.View(1):setNumInputDims(3))
	-- state size: 1	
	return netD
end
print("buiding model ......")
local Gmodel = get_G_model(opt)
Gmodel.name = 'AE'
Gmodel:apply(weights_init)

local Dmodel = get_D_model(opt)
Dmodel:apply(weights_init)

print("building model Done")

local D_criterion = nn.BCECriterion()			--the criterion of discriminator
local p_mse_criterion = nn.MSECriterion()	--pixel-wised mean square err
local p_gdl_criterin = nn.GDLCriterion(opt.gdlalpha) -- gdl loss function
local allCriterion = nn.MultiCriterion():add(p_mse_criterion, opt.pmseweight):add(p_gdl_criterin, opt.gdlweight)

-----------------------------------------------------

G_optimconfig = {
   learningRate = opt.lr,
   beta1 = opt.beta1,	
}

D_optimconfig ={
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

local real_label = 1
local fake_label = 0
local output_image = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local errrD, errG, outputG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
------------------------------------------------------
if opt.gpu > 0 then
	print("Using GPU")
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	input_image = input_image:cuda(); input_condition:cuda();output_image = output_image:cuda();label = label:cuda()
	if pcall(require, 'cudnn') then
		print("loading cudnn  model success")
		require 'cudnn'
		cudnn.benchmark = true
		cudnn.fastest = true 
		cudnn.convert(Dmodel, cudnn)
		cudnn.convert(Gmodel, cudnn)
	end
	Gmodel:cuda(); Dmodel:cuda(); allCriterion:cuda(); D_criterion:cuda()
end
----------------------------------------------------

local parametersD, gradparametersD = Dmodel:getParameters()
local  parametersG, gradparametersG = Gmodel:getParameters()
local imagebatch
----create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
	collectgarbage()
	gradparametersD:zero()
	if x~= parametersD then
		parametersD:copy(x)
	end

	-- train with real
	data_tm:reset()
	data_tm:resume()
	imagebatch = dataset:selectbatch()
	data_tm:stop()
	input_image:copy(imagebatch)
	label:fill(real_label)

	local  output = Dmodel:forward(input_image)
	local errD_real = D_criterion:forward(output, label)

	local df_do = D_criterion:backward(output, label)
	Dmodel:backward(input_image, df_do)

	--train with fake
	--input_image:copy(imagebatch)
	outputG = Gmodel:forward({input_image, input_condition})
	input_image:copy(outputG)
	label:fill(fake_label)

	output = Dmodel:forward(input_image)
	local errD_fake = D_criterion:forward(output, label)

	local df_do = D_criterion:backward(output, label)
	Dmodel:backward(input_image, df_do)

	errD = errD_real + errD_fake
	return errD, gradparametersD
end

---create closure to evaluate f(x) and df/dx of generator
local fGx = function (x)
	collectgarbage()
	if x~= parametersG then
		parametersG:copy(x) -- get new parameters
	end
	--print("parameters:sum=", parameters:sum())
	gradparametersG:zero() -- reset gradients

	---imagebatch has been loaded in fDx 
	
	output_image:copy(imagebatch)
	output_image:mul(2):add(-1)	--make output_image as (-1, 1)(tanh())
	--forward
	--local output = model:forward({input_image, input_condition})
	--outputG has been computed in fDx
	myerror = allCriterion:forward(outputG, output_image)

	--backward
	local d_p_g_cri_dG_output = allCriterion:backward(outputG, output_image)
	local derr_dG_output = d_p_g_cri_dG_output

	label:fill(fake_label) --fake labels are real fot generator cost

	local output = Dmodel.output -- netD:forward(input) was already executed in fDx, so save computation 
	errG = D_criterion:forward(output, label)

	local dD_cri_dD_output = D_criterion:backward(output, label)
	input_image:copy(outputG)
	local dD_cri_dG_output = Dmodel:updateGradInput(input_image, dD_cri_dD_output)
	dD_cri_dG_output:mul(opt.advweight)
	derr_dG_output:add(dD_cri_dG_output)

	input_image:copy(imagebatch)
	Gmodel:backward({input_image, input_condition}, derr_dG_output)

	allerr = myerror + errG * opt.advweight
	return allerr, gradparametersG
end

for epoch = 1, opt.niter do
	epoch_tm:reset()
	local counter = 0
	for i = 1, math.min(dataset:size(), opt.ntrain), opt.batchSize do --opt.train is useless now
		tm:reset()
		--must updata D network, because fGx use the D's result
		--debugger.enter()
		optim.adam(fDx, parametersD, D_optimconfig)

		optim.adam(fGx, parametersG, G_optimconfig)

	       print(('Epoch: [%d] \t Time: %.3f  DataTime: %.3f  '
	                   .. '  Err: %.4f  '):format(epoch, tm:time().real, data_tm:time().real, allerr))
	       xlua.progress(i, math.min(dataset:size(), opt.ntrain))
	end

	if epoch > 0 then
		paths.mkdir('gdl_gan_checkpoints')
		parametersD, gradparametersD= nil, nil --nil them to avoid spiking memory
		parametersG, gradparametersG=nil, nil
		torch.save('gdl_gan_checkpoints/'..opt.name..'_'..epoch..'netD.t7', Dmodel:clearState())
		torch.save('gdl_gan_checkpoints/'..opt.name..'_'..epoch..'netG.t7', Gmodel:clearState())

		parametersD, gradparametersD = Dmodel:getParameters()
		parametersG, gradparametersG = Gmodel:getParameters()
	end
   	print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
       	epoch, opt.niter, epoch_tm:time().real))	
   	xlua.progress(epoch, opt.niter)
end
os.execute('open -a  AE.svg')

