# api_app.py — local FastAPI server for the Attention-Guided DR grading model.
# Extracted from major_project.ipynb cell 35 so inference runs on this PC instead of Colab.
# Run:  .venv\Scripts\python api_app.py      (serves http://127.0.0.1:8080)
import os  # read the checkpoint path from an env var and check the file exists
import io  # wrap uploaded bytes in a file-like object PIL can open
import torch  # tensor math + loading the trained state_dict
import torch.nn as nn  # layer building blocks for the attention modules
import uvicorn  # ASGI server, so `python api_app.py` is the whole launch command
from fastapi import FastAPI, File, UploadFile  # HTTP API + multipart image upload
from fastapi.middleware.cors import CORSMiddleware  # the web UI calls this API from a different origin
from PIL import Image  # decode JPEG/PNG fundus images
import torchvision.transforms as transforms  # must mirror training preprocessing or predictions drift
import torchvision.models as models  # EfficientNet-B0 backbone the model was trained on


# --- Model architecture: must match the training notebook layer-for-layer, or load_state_dict fails ---
class ChannelAttention(nn.Module):  # CBAM channel attention: learns WHICH feature maps matter
    def __init__(self, in_planes, ratio=16):  # ratio 16 = bottleneck size used at training time
        super(ChannelAttention, self).__init__()  # register submodules with nn.Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average → one value per channel
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # global max catches small bright lesions avg-pool dilutes
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # squeeze; bias=False matches checkpoint keys
        self.relu1 = nn.ReLU()  # non-linearity inside the bottleneck
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # excite back to full channel count
        self.sigmoid = nn.Sigmoid()  # squash to 0..1 so it acts as a per-channel gate

    def forward(self, x):  # x: (batch, channels, h, w)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # shared MLP on the avg descriptor
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # same weights on the max descriptor
        out = avg_out + max_out  # fuse both views before gating
        return self.sigmoid(out)  # caller multiplies features by this gate


class SpatialAttention(nn.Module):  # CBAM spatial attention: learns WHERE in the retina to look
    def __init__(self, kernel_size=7):  # 7x7 = receptive field used at training time
        super(SpatialAttention, self).__init__()  # register submodules
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  # 2 maps in (avg,max) → 1 mask
        self.sigmoid = nn.Sigmoid()  # 0..1 mask per pixel

    def forward(self, x):  # x: (batch, channels, h, w)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # average across channels per pixel
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # strongest channel response per pixel
        x_cat = torch.cat([avg_out, max_out], dim=1)  # stack into the 2-channel conv input
        x_out = self.conv1(x_cat)  # learn the spatial mask
        return self.sigmoid(x_out)  # caller multiplies features by this mask


class GlobalAttentionBlock(nn.Module):  # channel-then-spatial attention, CBAM order
    def __init__(self, in_planes):  # in_planes = backbone output channels (1280)
        super(GlobalAttentionBlock, self).__init__()  # register submodules
        self.ca = ChannelAttention(in_planes)  # attribute name "ca" is part of the checkpoint keys
        self.sa = SpatialAttention()  # attribute name "sa" is part of the checkpoint keys

    def forward(self, x):  # refine backbone features
        x = x * self.ca(x)  # reweight channels first
        x = x * self.sa(x)  # then reweight locations
        return x  # same shape as input


class CategoryAttentionBlock(nn.Module):  # one attention map per DR grade
    def __init__(self, in_planes, num_classes=5):  # 5 grades: No DR → Proliferative
        super(CategoryAttentionBlock, self).__init__()  # register submodules
        self.num_classes = num_classes  # loop bound in forward
        self.conv = nn.Conv2d(in_planes, num_classes, 1)  # 1x1 conv → one map per class (bias kept, matches checkpoint)
        self.sigmoid = nn.Sigmoid()  # 0..1 masks

    def forward(self, x):  # x: refined features
        att_maps = self.sigmoid(self.conv(x))  # (batch, 5, h, w)
        out_features = []  # collects one pooled vector per class
        for i in range(self.num_classes):  # each grade pools features through its own mask
            mask = att_maps[:, i:i + 1, :, :]  # keep dim so it broadcasts over channels
            weighted = x * mask  # suppress regions irrelevant to this grade
            pooled = torch.mean(weighted, dim=[2, 3])  # (batch, 1280)
            out_features.append(pooled)  # gather for concat
        return torch.cat(out_features, dim=1), att_maps  # (batch, 1280*5) + maps for heatmaps


class AttentionGuidedCNN(nn.Module):  # full model: EfficientNet-B0 + global + category attention
    def __init__(self, num_classes=5, use_attention=True):  # defaults match the trained checkpoint
        super(AttentionGuidedCNN, self).__init__()  # register submodules
        self.use_attention = use_attention  # ablation switch kept so forward() matches training
        # weights=None: the checkpoint overwrites every backbone weight, so skip the 20 MB ImageNet download on startup
        backbone = models.efficientnet_b0(weights=None)  # same architecture as training, no network call
        self.features = backbone.features  # conv trunk only; classifier head replaced below
        in_planes = 1280  # EfficientNet-B0 final feature channels
        if use_attention:  # trained model uses attention
            self.global_att = GlobalAttentionBlock(in_planes)  # attribute names = checkpoint key prefixes
            self.category_att = CategoryAttentionBlock(in_planes, num_classes)  # per-grade pooling
            self.fc = nn.Linear(in_planes * num_classes, num_classes)  # 6400 → 5 logits
        else:  # baseline path, unused at serve time but keeps the class identical to training
            self.pool = nn.AdaptiveAvgPool2d(1)  # plain global pooling
            self.fc = nn.Linear(in_planes, num_classes)  # 1280 → 5 logits

    def forward(self, x):  # x: normalized (batch, 3, 384, 384)
        x = self.features(x)  # backbone features
        if self.use_attention:  # attention path
            x = self.global_att(x)  # CBAM refinement
            pooled_feats, att_maps = self.category_att(x)  # per-grade pooling
            out = self.fc(pooled_feats)  # class logits
            return out, att_maps  # maps returned for optional visualisation
        x = self.pool(x)  # baseline path
        x = torch.flatten(x, 1)  # (batch, 1280)
        return self.fc(x), None  # logits, no maps


# --- App setup ---
app = FastAPI(title="Diabetic Retinopathy Grading API")  # title shows in /docs

app.add_middleware(  # browser UI on another origin (ngrok URL / localhost dev server) must be allowed
    CORSMiddleware,  # Starlette CORS handler
    allow_origins=["*"],  # demo API, no auth or cookies, so any origin is fine
    allow_methods=["*"],  # GET / and POST /predict
    allow_headers=["*"],  # multipart content-type headers from the UI
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if a CUDA torch build is installed
model = AttentionGuidedCNN(num_classes=5, use_attention=True).to(device)  # build the empty model
# Env var lets you point at best_model_real_20epochs.pt without editing code; default sits next to this file
checkpoint_path = os.environ.get("DR_CHECKPOINT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_real.pt"))
weights_loaded = os.path.exists(checkpoint_path)  # surfaced in every response so random-weight output is never mistaken for a diagnosis

if weights_loaded:  # real trained model available
    # weights_only=True: a .pt is a pickle; this refuses arbitrary code in a file someone sent you
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"API loaded weights from {checkpoint_path}")  # startup log confirms which file is live
else:  # no checkpoint yet: server still runs so the UI/tunnel can be tested
    print(f"API WARNING: {checkpoint_path} not found. Predictions are RANDOM.")  # loud, because output looks real
model.eval()  # disable dropout/batchnorm updates for deterministic inference

transform_fn = transforms.Compose([  # identical to the notebook's validation transform
    transforms.Resize((384, 384)),  # training resolution
    transforms.ToTensor(),  # HWC uint8 → CHW float 0..1
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats the backbone expects
])

DR_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]  # index = predicted grade


@app.get("/")  # health check the UI / ngrok test can hit
def home():  # cheap liveness probe
    return {"message": "Attention-Guided DR Grading API is up and running!", "weights_loaded": weights_loaded}  # tells caller if output is meaningful


@app.post("/predict")  # main inference endpoint, field name "file"
async def predict(file: UploadFile = File(...)):  # multipart upload from the UI
    try:  # bad uploads return a JSON error instead of a 500
        contents = await file.read()  # whole image into memory; fundus JPEGs are a few MB
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # force 3 channels (PNG may be RGBA/greyscale)
        tensor = transform_fn(image).unsqueeze(0).to(device)  # add batch dim
        with torch.no_grad():  # no autograd graph: faster, less memory
            outputs, _ = model(tensor)  # logits, ignore attention maps
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()  # logits → probabilities
            prediction = torch.argmax(outputs, dim=1).item()  # most likely grade
        probs_dict = {DR_CLASSES[i]: float(probabilities[i]) for i in range(5)}  # JSON-friendly floats
        return {  # response shape kept identical to the Colab version so the UI needs no change
            "status": "success",  # UI branches on this
            "prediction_grade": prediction,  # 0..4
            "diagnosis": DR_CLASSES[prediction],  # human label
            "confidence_scores": probs_dict,  # per-class probabilities
            "weights_loaded": weights_loaded,  # false = random model, ignore the diagnosis
        }
    except Exception as e:  # corrupt image, wrong field name, etc.
        return {"status": "error", "message": str(e)}  # same error shape as the Colab version


if __name__ == "__main__":  # `python api_app.py` launches directly, no separate uvicorn command
    uvicorn.run(app, host="127.0.0.1", port=8080)  # loopback only: ngrok connects locally, LAN never sees it
