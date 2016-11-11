--learning temporal transformations from time-lapse videos, recurrent  generation

require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'xlua'
require 'paths'
require 'torch'
require('image')
require'mylstm'
debugger = require 'fb.debugger'
paths.dofile('GDLCriterion.lua')
paths.dofile('image_error_measures.lua')
debugger = require 'fb.debugger'
nngraph.setDebug(true)
--debugger.enter()
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
   name = 'recurrent_generator_restru',
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
   advweight = 0.2, -- the weight of adv loss function in generator model
   lstmInputsize = 512, -- the inputsize of lstm
   lstmOutputsize = 512, -- the outputsize of lstm
   length = 3,		--the length of timesteps
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

	local lstmInputsize = opt.lstmInputsize
	local lstmOutputsize = opt.lstmOutputsize

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

	encoder:add(nn.View(ndf * 8):setNumInputDims(3))
	--state size:(8 * ndf) x 1

	--input_image :RGB color image 3 * 64 *64
	local input_image = nn.Identity()() 
	--imgz: (8 * ndf)
	local imgz = encoder(input_image)		--input

	local lstm = mylstm(lstmInputsize, lstmOutputsize) --faster lstm
	--local x = torch.zeros(lstmInputsize)	             
	local prev_c = nn.Identity()()                                       --c initialize torch.zeros(lstmInputsize)
	local prev_h = nn.Identity()()                                       --h initialize 


	local lstmOutput1 = {imgz ,prev_c, prev_h} - lstm
	local lstmOutput2 = {imgz, lstmOutput1 - nn.SelectTable(1), lstmOutput1- nn.SelectTable(2)} - lstm
	local lstmOutput3 = {imgz, lstmOutput2 - nn.SelectTable(1), lstmOutput2 - nn.SelectTable(2)} - lstm


	local nde = ndf * 8
	--encoder input channel : nde = 8 * ndf = 512
	local decoder = nn.Sequential()
	decoder:add(SpatialFullConvolution(nde, ndf * 4, kW, kH, dW, dH))
	decoder:add(SpatialBatchNormalization(ndf * 4))
	decoder:add(nn.ReLU(true))
	--output size: 256, (1-1) * 2 + kW = 5
	decoder:add(SpatialFullConvolution(ndf * 4, ndf * 2, kW, kH, dW, dH))
	decoder:add(SpatialBatchNormalization(ndf * 2))
	decoder:add(nn.ReLU(true))
	--output size: 128, (5 - 1) * 2 + 5 = 13
	decoder:add(SpatialFullConvolution(ndf * 2, ndf , kW, kH, dW, dH, 0, 0, 1, 1))
	decoder:add(SpatialBatchNormalization(ndf))
	decoder:add(nn.ReLU(true))
	--output size: 64, (13 - 1) * 2 + 5 + 1 = 30
	decoder:add(SpatialFullConvolution(ndf, nc, kW, kH, dW, dH, 0, 0, 1, 1))
	decoder:add(nn.Tanh())
	--output size:3, (30 - 1) * 2 + 5 + 1 = 64
	--debugger.enter()
	local decoder_input1 = lstmOutput1- nn.SelectTable(2) - nn.View(ndf * 8, 1, 1)
	local decoder_input2 = lstmOutput2 - nn.SelectTable(2) - nn.View(ndf * 8, 1, 1)
	local decoder_input3 = lstmOutput3 - nn.SelectTable(2) - nn.View(ndf * 8, 1, 1)
	local decoder_img1 = decoder(decoder_input1)
	local decoder_img2 = decoder(decoder_input2)
	local decoder_img3 = decoder(decoder_input3)

	recurrent_generator = nn.gModule({input_image, prev_c, prev_h}, {decoder_img1, decoder_img2, decoder_img3})
	--debugger.enter()
	--graph.dot(recurrent_generator.fg, 'recurrent_generator forward', 'recurrent_generator forward')
	return recurrent_generator
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
Gmodel.name = 'recurrent_generator'
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
local prev_h = torch.zeros(opt.batchSize, opt.lstmInputsize) --initialize prev_h with 0
local prev_c = torch.zeros(opt.batchSize, opt.lstmInputsize)  --initializa prev_c with 0

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
	--debugger.enter()
	input_image = input_image:cuda(); prev_c = prev_c:cuda(); prev_h = prev_h:cuda()
	output_image = output_image:cuda()
	label = label:cuda()

	if pcall(require, 'cudnn') then
		print("Using cudnn  model success")
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
	--debugger.enter()
	outputG = Gmodel:forward({input_image, prev_c, prev_h})	-- three images of the generator outputs
	--[[
	true_img = torch.zeros(1, 3, 64, 64)--float Tensor
	pre_img = torch.zeros(1, 3, 64, 64)

	true_img:copy(input_image[1])
	true_img:resize(3, 64, 64)

	debugger.enter()
	a = image.rgb2y(true_img)
	pre_img:copy(outputG[1])
	pre_img:add(1):mul(0.5)
	pre_img:resize(3, 64, 64)
	image.display(pre_img)

	SSIMerr = SSIM(true_img, pre_img)
	PSNRerr = PSNR(true_img, pre_img)
	debugger.enter()
	]]--
	local errD_fake = 0
	label:fill(fake_label)
	allD_output = {}			-- reuse for optimizing G model
	for i=1, #outputG do           --length = #outputG
		input_image:copy(outputG[i])
		output = Dmodel:forward(input_image)
		allD_output[i] = output
		local errD_fake =  errD_fake + D_criterion:forward(output, label)
		local df_do = D_criterion:backward(output, label)
		Dmodel:backward(input_image, df_do)
	end

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

	local myerror = 0  --mse and gdl error 	
	local errG = 0
	--debugger.enter()
	label:fill(fake_label) --fake labels are real fot generator cost
	local derr_dG_output = {}
	for i = 1, #outputG do
		--forward
		--local output = model:forward({input_image, input_condition})
		--outputG has been computed in fDx
		myerror = myerror + allCriterion:forward(outputG[i], output_image)

		--backward
		local d_p_g_cri_dG_output = allCriterion:backward(outputG[i], output_image)
		derr_dG_output[i] = d_p_g_cri_dG_output		
		
		local output = allD_output[i] -- netD:forward(input) was already executed in fDx, so save computation 
		errG = errG + D_criterion:forward(output, label)
		local dD_cri_dD_output = D_criterion:backward(output, label)

		input_image:copy(outputG[i])
		local dD_cri_dG_output = Dmodel:updateGradInput(input_image, dD_cri_dD_output)

		dD_cri_dG_output:mul(opt.advweight)
		derr_dG_output[i]:add(dD_cri_dG_output)

	end	
	--debugger.enter()
	input_image:copy(imagebatch)
	Gmodel:backward({input_image, prev_c ,prev_h}, derr_dG_output)	
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

