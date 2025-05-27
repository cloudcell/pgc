# Cloudcell Ltd Presents...

## PGC version 0.1.0, codename "Dommekracht"

![logo_00_transparent_v2](https://github.com/user-attachments/assets/c8c4837a-5fcd-41e9-a2c5-6fba2cfc16d0)


# Code for Paper: "Polymorphic Graph Classifier"
### http://dx.doi.org/10.13140/RG.2.2.15744.55041

| Author |   Dates | License |
| ------ |   ---- | ------- |
| Alexander Bikeyev | 2025-04-17/present | AGPL v3 |

### Help
[Discord](https://discord.gg/daTSuB2z)

### Running the Example
```
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install uuencode
sudo apt-get install sharutils

# Start TensorBoard
tensorboard --logdir ./runs/ --port 6006

# Encode a file
python3 pcg.py any_file_to_pack

# Decode a file using standard header (98 '@'s)
python3 675_pgc_reader.py
 
```
### More Info:

https://github.com/cloudcell/prj.pgc-paper-code-public

---
# Research Log

2025-05-26 08:33

A key "load-bearing" pathway gets destroyed:
![20250526_0829_gorinnosho_collapse_at_top](https://github.com/user-attachments/assets/6b45ecbe-27d0-455a-8f0c-d24fbb34d44b)

![image](https://github.com/user-attachments/assets/f6165dfc-f769-4d34-9c76-6aa2e80f2b71)


2025-05-26 06:24

ResNet features turned out to be harmful to the performance. Demo -- below.

![resnet-collapse](https://github.com/user-attachments/assets/df55c256-22b6-40a4-9a35-d75603176adc)

This is called "ResNet Collapse" (green) vs "Normal Accuracy" (blue).

2025-05-16 02:43

Here's PGC learning MNIST-FASHION dataset (8 dimensional core with only 3 dimensions visible):

![20250516_0233_mnist-fashion-2st-attempt](https://github.com/user-attachments/assets/cd84bcdd-6043-4c16-ad98-94b7893e0f8a)


---
2025-05-04 11:19

Here's the dashboard used during training on the uu-encoded text "LICENSE" (from this repo), which it learns with 100% accuracy (!):

![image](https://github.com/user-attachments/assets/1711d0b5-147b-448e-af0d-adbea8b5dbaf)

Packed file 'LICENSE'

```
2025-05-04 11:19:04 - === Starting new text generation session ===
2025-05-04 11:19:23 - User selected option: 4
2025-05-04 11:19:32 - Set tokens to: 32000
2025-05-04 11:19:47 - User selected option: 5
2025-05-04 11:19:47 - Attempted to generate/unpack response without model
2025-05-04 11:19:49 - User selected option: 1
2025-05-04 11:19:50 - Selected checkpoint folder: checkpoints
2025-05-04 11:19:52 - Successfully loaded model from: model_20250504_080355_epoch_32.pt
2025-05-04 11:19:53 - User selected option: 5
2025-05-04 11:19:53 - Generating/unpacking response with input: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
2025-05-04 11:24:56 - Generated: <|sot|>@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@begin 664 LICENSE
M("`@("`@("`@("`@("`@("`@("!'3E4@049&15)/($=%3D5204P@4%5"3$E#
M($Q)0T5.4T4*("`@("`@("`@("`@("`@("`@("`@("!697)S:6]N(#,L(#$Y
M($YO=F5M8F5R(#(P,#<*"B!#;W!Y<FEG:'0@*$,I(#(P,#<@1G)E92!3;V9T
M=V%R92!&;W5N9&%T:6]N+"!);F,N(#QH='1P<SHO+V9S9BYO<F<O/@H@179E
M<GEO;F4@:7,@<&5R;6ET=&5D('1O(&-O<'D@86YD(&1I<W1R:6)U=&4@=F5R
M8F%T:6T@8V]P:65S"B!O9B!T:&ES(&QI8V5N<V4@9&]C=6UE;G0L(&)U="!C
M:&%N9VEN9R!I="!I<R!N;W0@86QL;W=E9"X*"B`@("`@("`@("`@("`@("`@
M("`@("`@("`@("!0<F5A;6)L90H*("!4:&4@1TY5($%F9F5R;R!'96YE<F%L
M(%!U8FQI8R!,:6-E;G-E(&ES(&$@9G)E92P@8V]P>6QE9G0@;&EC96YS92!F
M;W(*<V]F='=A<F4@86YD(&]T:&5R(&MI;F1S(&]F('=O<FMS+"!S<&5C:69I
M8V%L;'D@9&5S:6=N960@=&\@96YS=7)E"F-O;W!E<F%T:6]N('=I=&@@=&AE
M(&-O;6UU;FET>2!I;B!T:&4@8V%S92!O9B!N971W;W)K('-E<G9E<B!S;V9T
M=V%R92X*"B`@5&AE(&QI8V5N<V5S(&9O<B!M;W-T('-O9G1W87)E(&%N9"!O
M=&AE<B!P<F%C=&EC86P@=V]R:W,@87)E(&1E<VEG;F5D"G1O('1A:V4@87=A
M>2!Y;W5R(&9R965D;VT@=&\@<VAA<F4@86YD(&-H86YG92!T:&4@=V]R:W,N
M("!">2!C;VYT<F%S="P*;W5R($=E;F5R86P@4'5B;&EC($QI8V5N<V5S(&%R
M92!I;G1E;F1E9"!T;R!G=6%R86YT964@>6]U<B!F<F5E9&]M('1O"G-H87)E
M(&%N9"!C:&%N9V4@86QL('9E<G-I;VYS(&]F(&$@<')O9W)A;2TM=&\@;6%K
M92!S=7)E(&ET(')E;6%I;G,@9G)E90IS;V9T=V%R92!F;W(@86QL(&ET<R!U
M<V5R<RX*"B`@5VAE;B!W92!S<&5A:R!O9B!F<F5E('-O9G1W87)E+"!W92!A
M<F4@<F5F97)R:6YG('1O(&9R965D;VTL(&YO=`IP<FEC92X@($]U<B!'96YE
M<F%L(%!U8FQI8R!,:6-E;G-E<R!A<F4@9&5S:6=N960@=&\@;6%K92!S=7)E
M('1H870@>6]U"FAA=F4@=&AE(&9R965D;VT@=&\@9&ES=')I8G5T92!C;W!I
M97,@;V8@9G)E92!S;V9T=V%R92`H86YD(&-H87)G92!F;W(*=&AE;2!I9B!Y
M;W4@=VES:"DL('1H870@>6]U(')E8V5I=F4@<V]U<F-E(&-O9&4@;W(@8V%N
M(&=E="!I="!I9B!Y;W4*=V%N="!I="P@=&AA="!Y;W4@8V%N(&-H86YG92!T
M:&4@<V]F='=A<F4@;W(@=7-E('!I96-E<R!O9B!I="!I;B!N97<*9G)E92!P
M<F]G<F%M<RP@86YD('1H870@>6]U(&MN;W<@>6]U(&-A;B!D;R!T:&5S92!T
M:&EN9W,N"@H@($1E=F5L;W!E<G,@=&AA="!U<V4@;W5R($=E;F5R86P@4'5B
M;&EC($QI8V5N<V5S('!R;W1E8W0@>6]U<B!R:6=H=',*=VET:"!T=V\@<W1E
M<',Z("@Q*2!A<W-E<G0@8V]P>7)I9VAT(&]N('1H92!S;V9T=V%R92P@86YD
M("@R*2!O9F9E<@IY;W4@=&AI<R!,:6-E;G-E('=H:6-H(&=I=F5S('EO=2!L
M96=A;"!P97)M:7-S:6]N('1O(&-O<'DL(&1I<W1R:6)U=&4*86YD+V]R(&UO
M9&EF>2!T:&4@<V]F='=A<F4N"@H@($$@<V5C;VYD87)Y(&)E;F5F:70@;V8@
M9&5F96YD:6YG(&%L;"!U<V5R<R<@9G)E961O;2!I<R!T:&%T"FEM<')O=F5M
M96YT<R!M861E(&EN(&%L=&5R;F%T92!V97)S:6]N<R!O9B!T:&4@<')O9W)A
M;2P@:68@=&AE>0IR96-E:79E('=I9&5S<')E860@=7-E+"!B96-O;64@879A
M:6QA8FQE(&9O<B!O=&AE<B!D979E;&]P97)S('1O"FEN8V]R<&]R871E+B`@
M36%N>2!D979E;&]P97)S(&]F(&9R964@<V]F='=A<F4@87)E(&AE87)T96YE
...
...
...
<|eot|>
2025-05-04 11:46:00 - Generation stats - Model calls: 128000, Time: 1232.40s, Speed: 103.86 calls/s

```



