??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
#ActorNetwork/input_mlp/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#ActorNetwork/input_mlp/dense/kernel
?
7ActorNetwork/input_mlp/dense/kernel/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/kernel* 
_output_shapes
:
??*
dtype0
?
!ActorNetwork/input_mlp/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!ActorNetwork/input_mlp/dense/bias
?
5ActorNetwork/input_mlp/dense/bias/Read/ReadVariableOpReadVariableOp!ActorNetwork/input_mlp/dense/bias*
_output_shapes	
:?*
dtype0
?
%ActorNetwork/input_mlp/dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%ActorNetwork/input_mlp/dense/kernel_1
?
9ActorNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOpReadVariableOp%ActorNetwork/input_mlp/dense/kernel_1* 
_output_shapes
:
??*
dtype0
?
#ActorNetwork/input_mlp/dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#ActorNetwork/input_mlp/dense/bias_1
?
7ActorNetwork/input_mlp/dense/bias_1/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/bias_1*
_output_shapes	
:?*
dtype0
?
%ActorNetwork/input_mlp/dense/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%ActorNetwork/input_mlp/dense/kernel_2
?
9ActorNetwork/input_mlp/dense/kernel_2/Read/ReadVariableOpReadVariableOp%ActorNetwork/input_mlp/dense/kernel_2* 
_output_shapes
:
??*
dtype0
?
#ActorNetwork/input_mlp/dense/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#ActorNetwork/input_mlp/dense/bias_2
?
7ActorNetwork/input_mlp/dense/bias_2/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/bias_2*
_output_shapes	
:?*
dtype0
?
ActorNetwork/action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameActorNetwork/action/kernel
?
.ActorNetwork/action/kernel/Read/ReadVariableOpReadVariableOpActorNetwork/action/kernel* 
_output_shapes
:
??*
dtype0
?
ActorNetwork/action/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameActorNetwork/action/bias
?
,ActorNetwork/action/bias/Read/ReadVariableOpReadVariableOpActorNetwork/action/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
	3

4
5
6
7

0
 
ec
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE!ActorNetwork/input_mlp/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%ActorNetwork/input_mlp/dense/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%ActorNetwork/input_mlp/dense/kernel_2,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/bias_2,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEActorNetwork/action/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEActorNetwork/action/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE

ref
1

_actor_network
c
_mlp_layers
	variables
trainable_variables
regularization_losses
	keras_api
#
0
1
2
3
4
8
0
1
2
	3

4
5
6
7
8
0
1
2
	3

4
5
6
7
 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

kernel
bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

kernel
	bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h


kernel
bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

kernel
bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
 
#
0
1
2
3
4
 
 
 
 
 
 
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
 	variables
!trainable_variables
"regularization_losses

0
1

0
1
 
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
$	variables
%trainable_variables
&regularization_losses

0
	1

0
	1
 
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
(	variables
)trainable_variables
*regularization_losses


0
1


0
1
 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
,	variables
-trainable_variables
.regularization_losses

0
1

0
1
 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
0	variables
1trainable_variables
2regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0_discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
y
action_0_observationPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
j
action_0_rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0_step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/bias%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1%ActorNetwork/input_mlp/dense/kernel_2#ActorNetwork/input_mlp/dense/bias_2ActorNetwork/action/kernelActorNetwork/action/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_277836
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_277841
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_277853
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_277849
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp7ActorNetwork/input_mlp/dense/kernel/Read/ReadVariableOp5ActorNetwork/input_mlp/dense/bias/Read/ReadVariableOp9ActorNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOp7ActorNetwork/input_mlp/dense/bias_1/Read/ReadVariableOp9ActorNetwork/input_mlp/dense/kernel_2/Read/ReadVariableOp7ActorNetwork/input_mlp/dense/bias_2/Read/ReadVariableOp.ActorNetwork/action/kernel/Read/ReadVariableOp,ActorNetwork/action/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_277908
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/bias%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1%ActorNetwork/input_mlp/dense/kernel_2#ActorNetwork/input_mlp/dense/bias_2ActorNetwork/action/kernelActorNetwork/action/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_277945ٮ
?r
?	
'__inference_polymorphic_action_fn_23823
	step_type

reward
discount
observationO
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
??K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	?F
2actornetwork_action_matmul_readvariableop_resource:
??B
3actornetwork_action_biasadd_readvariableop_resource:	?
identity??*ActorNetwork/action/BiasAdd/ReadVariableOp?)ActorNetwork/action/MatMul/ReadVariableOp?3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp?2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOph
ActorNetwork/CastCastobservation*

DstT0*

SrcT0*(
_output_shapes
:??????????k
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:???????????
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   U  ?
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:??????????w
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:??????????[
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB 2        [
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        r
+Deterministic/mode/Deterministic/mean/ShapeShapeActorNetwork/Cast_1:y:0*
T0*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:???????????????????]
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
valueB 2        ]
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*9
_output_shapes'
%:#???????????????????y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*5
_output_shapes#
!:???????????????????`
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*5
_output_shapes#
!:???????????????????X
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*5
_output_shapes#
!:???????????????????n
IdentityIdentityclip_by_value:z:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:?????????:?????????:?????????:??????????: : : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:UQ
(
_output_shapes
:??????????
%
_user_specified_nameobservation
?r
?	
'__inference_polymorphic_action_fn_23654
	time_step
time_step_1
time_step_2
time_step_3O
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
??K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	?F
2actornetwork_action_matmul_readvariableop_resource:
??B
3actornetwork_action_biasadd_readvariableop_resource:	?
identity??*ActorNetwork/action/BiasAdd/ReadVariableOp?)ActorNetwork/action/MatMul/ReadVariableOp?3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp?2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOph
ActorNetwork/CastCasttime_step_3*

DstT0*

SrcT0*(
_output_shapes
:??????????k
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:???????????
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   U  ?
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:??????????w
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:??????????[
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB 2        [
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        r
+Deterministic/mode/Deterministic/mean/ShapeShapeActorNetwork/Cast_1:y:0*
T0*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:???????????????????]
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
valueB 2        ]
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*9
_output_shapes'
%:#???????????????????y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*5
_output_shapes#
!:???????????????????`
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*5
_output_shapes#
!:???????????????????X
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*5
_output_shapes#
!:???????????????????n
IdentityIdentityclip_by_value:z:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:?????????:?????????:?????????:??????????: : : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:SO
(
_output_shapes
:??????????
#
_user_specified_name	time_step
?
6
$__inference_signature_wrapper_277841

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *2
f-R+
)__inference_function_with_signature_23706*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?r
?	
'__inference_polymorphic_action_fn_23912
time_step_step_type
time_step_reward
time_step_discount
time_step_observationO
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
??K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	?F
2actornetwork_action_matmul_readvariableop_resource:
??B
3actornetwork_action_biasadd_readvariableop_resource:	?
identity??*ActorNetwork/action/BiasAdd/ReadVariableOp?)ActorNetwork/action/MatMul/ReadVariableOp?3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp?2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpr
ActorNetwork/CastCasttime_step_observation*

DstT0*

SrcT0*(
_output_shapes
:??????????k
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:???????????
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   U  ?
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:??????????w
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:??????????[
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB 2        [
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        r
+Deterministic/mode/Deterministic/mean/ShapeShapeActorNetwork/Cast_1:y:0*
T0*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:???????????????????]
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
valueB 2        ]
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Deterministic_1/sample/ShapeShape:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0*
T0*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*9
_output_shapes'
%:#???????????????????y
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*5
_output_shapes#
!:???????????????????`
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*5
_output_shapes#
!:???????????????????X
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB 2      ???
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*5
_output_shapes#
!:???????????????????n
IdentityIdentityclip_by_value:z:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:?????????:?????????:?????????:??????????: : : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:_[
(
_output_shapes
:??????????
/
_user_specified_nametime_step/observation
? 
?
__inference__traced_save_277908
file_prefix'
#savev2_variable_read_readvariableop	B
>savev2_actornetwork_input_mlp_dense_kernel_read_readvariableop@
<savev2_actornetwork_input_mlp_dense_bias_read_readvariableopD
@savev2_actornetwork_input_mlp_dense_kernel_1_read_readvariableopB
>savev2_actornetwork_input_mlp_dense_bias_1_read_readvariableopD
@savev2_actornetwork_input_mlp_dense_kernel_2_read_readvariableopB
>savev2_actornetwork_input_mlp_dense_bias_2_read_readvariableop9
5savev2_actornetwork_action_kernel_read_readvariableop7
3savev2_actornetwork_action_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*?
value?B?
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop>savev2_actornetwork_input_mlp_dense_kernel_read_readvariableop<savev2_actornetwork_input_mlp_dense_bias_read_readvariableop@savev2_actornetwork_input_mlp_dense_kernel_1_read_readvariableop>savev2_actornetwork_input_mlp_dense_bias_1_read_readvariableop@savev2_actornetwork_input_mlp_dense_kernel_2_read_readvariableop>savev2_actornetwork_input_mlp_dense_bias_2_read_readvariableop5savev2_actornetwork_action_kernel_read_readvariableop3savev2_actornetwork_action_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*e
_input_shapesT
R: : :
??:?:
??:?:
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:


_output_shapes
: 
?
5
#__inference_get_initial_state_23979

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
5
#__inference_get_initial_state_23705

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
)__inference_function_with_signature_23673
	step_type

reward
discount
observation
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_polymorphic_action_fn_23654}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:?????????:?????????:?????????:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:WS
(
_output_shapes
:??????????
'
_user_specified_name0/observation
?
;
)__inference_function_with_signature_23706

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_get_initial_state_23705*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
d
$__inference_signature_wrapper_277849
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *2
f-R+
)__inference_function_with_signature_23718^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
+
)__inference_function_with_signature_23729?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *!
fR
__inference_<lambda>_690*(
_construction_contextkEagerRuntime*
_input_shapes 
?)
?
"__inference__traced_restore_277945
file_prefix#
assignvariableop_variable:	 J
6assignvariableop_1_actornetwork_input_mlp_dense_kernel:
??C
4assignvariableop_2_actornetwork_input_mlp_dense_bias:	?L
8assignvariableop_3_actornetwork_input_mlp_dense_kernel_1:
??E
6assignvariableop_4_actornetwork_input_mlp_dense_bias_1:	?L
8assignvariableop_5_actornetwork_input_mlp_dense_kernel_2:
??E
6assignvariableop_6_actornetwork_input_mlp_dense_bias_2:	?A
-assignvariableop_7_actornetwork_action_kernel:
??:
+assignvariableop_8_actornetwork_action_bias:	?
identity_10??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*?
value?B?
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp6assignvariableop_1_actornetwork_input_mlp_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_actornetwork_input_mlp_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp8assignvariableop_3_actornetwork_input_mlp_dense_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp6assignvariableop_4_actornetwork_input_mlp_dense_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_actornetwork_input_mlp_dense_kernel_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_actornetwork_input_mlp_dense_bias_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_actornetwork_action_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_actornetwork_action_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: V
Identity_10IdentityIdentity_9:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 "#
identity_10Identity_10:output:0*'
_input_shapes
: : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
i
)__inference_function_with_signature_23718
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *!
fR
__inference_<lambda>_687^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
_
__inference_<lambda>_687!
readvariableop_resource:	 
identity	??ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
?
&
$__inference_signature_wrapper_277853?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *2
f-R+
)__inference_function_with_signature_23729*(
_construction_contextkEagerRuntime*
_input_shapes 
?T
?	
-__inference_polymorphic_distribution_fn_23976
	step_type

reward
discount
observationO
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
??K
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	?Q
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
??M
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	?F
2actornetwork_action_matmul_readvariableop_resource:
??B
3actornetwork_action_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??*ActorNetwork/action/BiasAdd/ReadVariableOp?)ActorNetwork/action/MatMul/ReadVariableOp?3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp?5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp?2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp?4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOph
ActorNetwork/CastCastobservation*

DstT0*

SrcT0*(
_output_shapes
:??????????k
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:???????????
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   U  ?
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:??????????W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:??????????w
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:??????????[
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB 2        [
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        r
+Deterministic/mode/Deterministic/mean/ShapeShapeActorNetwork/Cast_1:y:0*
T0*
_output_shapes
:m
+Deterministic/mode/Deterministic/mean/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Deterministic/mode/Deterministic/mean/strided_sliceStridedSlice4Deterministic/mode/Deterministic/mean/Shape:output:0BDeterministic/mode/Deterministic/mean/strided_slice/stack:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_1:output:0DDeterministic/mode/Deterministic/mean/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
6Deterministic/mode/Deterministic/mean/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB {
8Deterministic/mode/Deterministic/mean/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
3Deterministic/mode/Deterministic/mean/BroadcastArgsBroadcastArgsADeterministic/mode/Deterministic/mean/BroadcastArgs/s0_1:output:0<Deterministic/mode/Deterministic/mean/strided_slice:output:0*
_output_shapes
:x
5Deterministic/mode/Deterministic/mean/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB s
1Deterministic/mode/Deterministic/mean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:???????????????????]
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
valueB 2        ]
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
valueB 2        [
IdentityIdentityDeterministic_1/atol:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_1Identity:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????]

Identity_2IdentityDeterministic_1/rtol:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:?????????:?????????:?????????:??????????: : : : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:UQ
(
_output_shapes
:??????????
%
_user_specified_nameobservation
Y

__inference_<lambda>_690*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
$__inference_signature_wrapper_277836
discount
observation

reward
	step_type
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *2
f-R+
)__inference_function_with_signature_23673}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:?????????:??????????:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:WS
(
_output_shapes
:??????????
'
_user_specified_name0/observation:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0_discount:0?????????
?
0/observation.
action_0_observation:0??????????
0
0/reward$
action_0_reward:0?????????
6
0/step_type'
action_0_step_type:0?????????H
action>
StatefulPartitionedCall:0???????????????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:?_
?

train_step
metadata
model_variables
_all_assets

signatures

Maction
Ndistribution
Oget_initial_state
Pget_metadata
Qget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
Y
0
1
2
	3

4
5
6
7"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

Raction
Sget_initial_state
Tget_train_step
Uget_metadata"
signature_map
9:7
?? 2#ActorNetwork/input_mlp/dense/kernel
2:0? 2!ActorNetwork/input_mlp/dense/bias
9:7
?? 2#ActorNetwork/input_mlp/dense/kernel
2:0? 2!ActorNetwork/input_mlp/dense/bias
9:7
?? 2#ActorNetwork/input_mlp/dense/kernel
2:0? 2!ActorNetwork/input_mlp/dense/bias
0:.
?? 2ActorNetwork/action/kernel
):'? 2ActorNetwork/action/bias
1
ref
1"
trackable_tuple_wrapper
2
_actor_network"
_generic_user_object
?
_mlp_layers
	variables
trainable_variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
	bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
?


kernel
bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
 	variables
!trainable_variables
"regularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
$	variables
%trainable_variables
&regularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
(	variables
)trainable_variables
*regularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
,	variables
-trainable_variables
.regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
0	variables
1trainable_variables
2regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
'__inference_polymorphic_action_fn_23823
'__inference_polymorphic_action_fn_23912?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_polymorphic_distribution_fn_23976?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_get_initial_state_23979?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_690"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_687"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_277836
0/discount0/observation0/reward0/step_type"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_277841
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_277849"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_277853"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecM
argsE?B
jself
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?
? 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecM
argsE?B
jself
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?
? 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 7
__inference_<lambda>_687?

? 
? "? 	0
__inference_<lambda>_690?

? 
? "? P
#__inference_get_initial_state_23979)"?
?
?

batch_size 
? "? ?
'__inference_polymorphic_action_fn_23823?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????5
observation&?#
observation??????????
? 
? "d?a

PolicyStep8
action.?+
action???????????????????
state? 
info? ?
'__inference_polymorphic_action_fn_23912?	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount??????????
observation0?-
time_step/observation??????????
? 
? "d?a

PolicyStep8
action.?+
action???????????????????
state? 
info? ?
-__inference_polymorphic_distribution_fn_23976?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????5
observation&?#
observation??????????
? 
? "???

PolicyStep?
action??????
`
T?Q

atol? 
-
loc&?#???????????????????

rtol? 
L?I

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
?
j
parameters
? 
?
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec 
state? 
info? ?
$__inference_signature_wrapper_277836?	
???
? 
???
.

0/discount ?

0/discount?????????
9
0/observation(?%
0/observation??????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"=?:
8
action.?+
action???????????????????_
$__inference_signature_wrapper_27784170?-
? 
&?#
!

batch_size?

batch_size "? X
$__inference_signature_wrapper_2778490?

? 
? "?

int64?
int64 	<
$__inference_signature_wrapper_277853?

? 
? "? 