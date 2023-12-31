import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_chn=3, out_chn=3, kernel_size=2, stride=2, BN_momentum=0.1, dropout_p = 0.2, padding = 2):
        super(UNet, self).__init__()

        #UNet
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively  + Dropout
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively + Dropout

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.enc_outputs = [] #save the downsampling stage layer

        self.MaxEn = nn.MaxPool2d(kernel_size=2, stride=2, padding = padding, return_indices=True) 
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.ConvEn11 = nn.Conv2d(self.in_chn, 64*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual11 = nn.Conv2d(self.in_chn, 64*2, kernel_size=1)
        self.BNEn11 = nn.BatchNorm2d(64*2, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual12 = nn.Conv2d(64, 64*2, kernel_size=1)
        self.BNEn12 = nn.BatchNorm2d(64*2, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual21 = nn.Conv2d(64, 128*2, kernel_size=1)
        self.BNEn21 = nn.BatchNorm2d(128*2, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual22 = nn.Conv2d(128, 128*2, kernel_size=1)
        self.BNEn22 = nn.BatchNorm2d(128*2, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual31 = nn.Conv2d(128, 256*2, kernel_size=1)
        self.BNEn31 = nn.BatchNorm2d(256*2, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual32 = nn.Conv2d(256, 256*2, kernel_size=1)
        self.BNEn32 = nn.BatchNorm2d(256*2, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual33 = nn.Conv2d(256, 256*2, kernel_size=1)
        self.BNEn33 = nn.BatchNorm2d(256*2, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual41 = nn.Conv2d(256, 512*2, kernel_size=1)
        self.BNEn41 = nn.BatchNorm2d(512*2, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual42 = nn.Conv2d(512, 512*2, kernel_size=1)
        self.BNEn42 = nn.BatchNorm2d(512*2, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.Residual43 = nn.Conv2d(512, 512*2, kernel_size=1)
        self.BNEn43 = nn.BatchNorm2d(512*2, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(kernel_size = 2, stride=2, padding = padding) 

        self.ConvDe43 = nn.Conv2d(512*2, 512*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe43 = nn.Conv2d(512*2, 512*2, kernel_size=1)
        self.BNDe43 = nn.BatchNorm2d(512*2, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe42 = nn.Conv2d(512, 512*2, kernel_size=1)
        self.BNDe42 = nn.BatchNorm2d(512*2, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe41 = nn.Conv2d(512, 256*2, kernel_size=1)
        self.BNDe41 = nn.BatchNorm2d(256*2, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256*2, 256*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe33 = nn.Conv2d(256*2, 256*2, kernel_size=1)
        self.BNDe33 = nn.BatchNorm2d(256*2, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe32 = nn.Conv2d(256, 256*2, kernel_size=1)
        self.BNDe32 = nn.BatchNorm2d(256*2, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe31 = nn.Conv2d(256, 128*2, kernel_size=1)
        self.BNDe31 = nn.BatchNorm2d(128*2, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128*2, 128*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe22 = nn.Conv2d(128*2, 128*2, kernel_size=1)
        self.BNDe22 = nn.BatchNorm2d(128*2, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe21 = nn.Conv2d(128, 64*2, kernel_size=1)
        self.BNDe21 = nn.BatchNorm2d(64*2, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64*2, 64*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe12 = nn.Conv2d(64*2, 64*2, kernel_size=1)
        self.BNDe12 = nn.BatchNorm2d(64*2, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn*2, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = 'reflect')
        self.ResidualDe11 = nn.Conv2d(64, self.out_chn*2, kernel_size=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn*2, momentum=BN_momentum)


        

    def forward(self, x):

        #ENCODE LAYERS
        #1
        residual = self.Residual11(x)
        x = F.glu(self.BNEn11(self.ConvEn11(x))+residual, dim=1)
        residual = self.Residual12(x)
        x = F.glu(self.BNEn12(self.ConvEn12(x))+residual, dim=1)
        self.enc_outputs.append(x)
        size1 = x.size()
        
        x, ind1 = self.MaxEn(x)
        x = self.dropout(x)

        #2
        residual = self.Residual21(x)
        x = F.glu(self.BNEn21(self.ConvEn21(x))+residual, dim=1)
        residual = self.Residual22(x)
        x = F.glu(self.BNEn22(self.ConvEn22(x))+residual, dim=1) 
        self.enc_outputs.append(x)
        size2 = x.size()
        
        x, ind2 = self.MaxEn(x)
        x = self.dropout(x)
        

        #3
        residual = self.Residual31(x)
        x = F.glu(self.BNEn31(self.ConvEn31(x))+residual, dim=1)
        residual = self.Residual32(x)
        x = F.glu(self.BNEn32(self.ConvEn32(x))+residual, dim=1)
        residual = self.Residual33(x)
        x = F.glu(self.BNEn33(self.ConvEn33(x))+residual, dim=1)   
        self.enc_outputs.append(x)
        size3 = x.size()
        
        x, ind3 = self.MaxEn(x)
        x = self.dropout(x)

        #4
        residual = self.Residual41(x)
        x = F.glu(self.BNEn41(self.ConvEn41(x))+residual, dim=1)
        residual = self.Residual42(x)
        x = F.glu(self.BNEn42(self.ConvEn42(x))+residual, dim=1)
        residual = self.Residual43(x)
        x = F.glu(self.BNEn43(self.ConvEn43(x))+residual, dim=1)   
        self.enc_outputs.append(x)
        size4 = x.size()
        
        # x, ind4 = self.MaxEn(x)

        #5
        # x = self.MaxDe(x, ind4, output_size=size3)
        x = self.dropout(x)
        x = torch.cat([x, self.enc_outputs.pop()], dim=1)
        residual = self.ResidualDe43(x)
        x = F.glu(self.BNDe43(self.ConvDe43(x))+residual, dim=1)
        residual = self.ResidualDe42(x)
        x = F.glu(self.BNDe42(self.ConvDe42(x))+residual, dim=1)
        residual = self.ResidualDe41(x)
        x = F.glu(self.BNDe41(self.ConvDe41(x))+residual, dim=1)

        #6
        x = self.MaxDe(x, ind3, output_size=size3)
        x = self.dropout(x)
        x = torch.cat([x, self.enc_outputs.pop()], dim=1)
        residual = self.ResidualDe33(x)
        x = F.glu(self.BNDe33(self.ConvDe33(x))+residual, dim=1)
        residual = self.ResidualDe32(x)
        x = F.glu(self.BNDe32(self.ConvDe32(x))+residual, dim=1)
        residual = self.ResidualDe31(x)
        x = F.glu(self.BNDe31(self.ConvDe31(x))+residual, dim=1)

        #7
        x = self.MaxDe(x, ind2, output_size=size2)
        x = self.dropout(x)
        x = torch.cat([x, self.enc_outputs.pop()], dim=1)
        residual = self.ResidualDe22(x)
        x = F.glu(self.BNDe22(self.ConvDe22(x))+residual, dim=1)
        residual = self.ResidualDe21(x)
        x = F.glu(self.BNDe21(self.ConvDe21(x))+residual, dim=1)

        #8
        x = self.MaxDe(x, ind1, output_size=size1)
        x = self.dropout(x)
        x = torch.cat([x, self.enc_outputs.pop()], dim=1)
        residual = self.ResidualDe12(x)
        x = F.glu(self.BNDe12(self.ConvDe12(x))+residual, dim=1)
        residual = self.ResidualDe11(x)
        x = F.glu(self.BNDe11(self.ConvDe11(x))+residual, dim=1)

        return x


