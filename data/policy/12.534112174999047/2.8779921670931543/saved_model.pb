‘ф
Ґу
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
delete_old_dirsbool(И
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
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
≥
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28йч
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
§
#ActorNetwork/input_mlp/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДА*4
shared_name%#ActorNetwork/input_mlp/dense/kernel
Э
7ActorNetwork/input_mlp/dense/kernel/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/kernel* 
_output_shapes
:
ДА*
dtype0
Ы
!ActorNetwork/input_mlp/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!ActorNetwork/input_mlp/dense/bias
Ф
5ActorNetwork/input_mlp/dense/bias/Read/ReadVariableOpReadVariableOp!ActorNetwork/input_mlp/dense/bias*
_output_shapes	
:А*
dtype0
®
%ActorNetwork/input_mlp/dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*6
shared_name'%ActorNetwork/input_mlp/dense/kernel_1
°
9ActorNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOpReadVariableOp%ActorNetwork/input_mlp/dense/kernel_1* 
_output_shapes
:
АА*
dtype0
Я
#ActorNetwork/input_mlp/dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#ActorNetwork/input_mlp/dense/bias_1
Ш
7ActorNetwork/input_mlp/dense/bias_1/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/bias_1*
_output_shapes	
:А*
dtype0
®
%ActorNetwork/input_mlp/dense/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*6
shared_name'%ActorNetwork/input_mlp/dense/kernel_2
°
9ActorNetwork/input_mlp/dense/kernel_2/Read/ReadVariableOpReadVariableOp%ActorNetwork/input_mlp/dense/kernel_2* 
_output_shapes
:
АА*
dtype0
Я
#ActorNetwork/input_mlp/dense/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#ActorNetwork/input_mlp/dense/bias_2
Ш
7ActorNetwork/input_mlp/dense/bias_2/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/bias_2*
_output_shapes	
:А*
dtype0
Т
ActorNetwork/action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АЂ*+
shared_nameActorNetwork/action/kernel
Л
.ActorNetwork/action/kernel/Read/ReadVariableOpReadVariableOpActorNetwork/action/kernel* 
_output_shapes
:
АЂ*
dtype0
Й
ActorNetwork/action/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ђ*)
shared_nameActorNetwork/action/bias
В
,ActorNetwork/action/bias/Read/ReadVariableOpReadVariableOpActorNetwork/action/bias*
_output_shapes	
:Ђ*
dtype0

NoOpNoOp
Т
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ќ
value√Bј Bє
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
≠
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
≠
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
≠
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
≠
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
≠
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
≠
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
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
y
action_0_observationPlaceholder*(
_output_shapes
:€€€€€€€€€Д*
dtype0*
shape:€€€€€€€€€Д
j
action_0_rewardPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
m
action_0_step_typePlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
°
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/bias%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1%ActorNetwork/input_mlp/dense/kernel_2#ActorNetwork/input_mlp/dense/bias_2ActorNetwork/action/kernelActorNetwork/action/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_360109
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ь
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
GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_360114
Ё
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
GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_360126
Ш
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
GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_360122
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ы
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_360181
¬
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_360218ўЃ
І 
£
__inference__traced_save_360181
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

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: °
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0* 
valueјBљ
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B ћ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop>savev2_actornetwork_input_mlp_dense_kernel_read_readvariableop<savev2_actornetwork_input_mlp_dense_bias_read_readvariableop@savev2_actornetwork_input_mlp_dense_kernel_1_read_readvariableop>savev2_actornetwork_input_mlp_dense_bias_1_read_readvariableop@savev2_actornetwork_input_mlp_dense_kernel_2_read_readvariableop>savev2_actornetwork_input_mlp_dense_bias_2_read_readvariableop5savev2_actornetwork_action_kernel_read_readvariableop3savev2_actornetwork_action_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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
ДА:А:
АА:А:
АА:А:
АЂ:Ђ: 2(
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
ДА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АЂ:!	

_output_shapes	
:Ђ:


_output_shapes
: 
пT
™	
-__inference_polymorphic_distribution_fn_24039
	step_type

reward
discount
observationO
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
ДАK
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	АF
2actornetwork_action_matmul_readvariableop_resource:
АЂB
3actornetwork_action_biasadd_readvariableop_resource:	Ђ
identity

identity_1

identity_2ИҐ*ActorNetwork/action/BiasAdd/ReadVariableOpҐ)ActorNetwork/action/MatMul/ReadVariableOpҐ3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpҐ2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOph
ActorNetwork/CastCastobservation*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Дk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€Д  Ц
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Д∞
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
ДА*
dtype0√
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А≠
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ќ
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЛ
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0—
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype0”
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЮ
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
АЂ*
dtype0љ
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЂЫ
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:Ђ*
dtype0≥
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ђy
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ђo
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   +  Щ
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?К
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Г
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Ђw
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:€€€€€€€€€Ђ[
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
value	B : Г
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
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
valueB б
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
value	B : Ь
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:–
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ]
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
: Щ

Identity_1Identity:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ]

Identity_2IdentityDeterministic_1/rtol:output:0^NoOp*
T0*
_output_shapes
: и
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€Д: : : : : : : : 2X
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
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:UQ
(
_output_shapes
:€€€€€€€€€Д
%
_user_specified_nameobservation
Юr
Д	
'__inference_polymorphic_action_fn_23886
	step_type

reward
discount
observationO
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
ДАK
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	АF
2actornetwork_action_matmul_readvariableop_resource:
АЂB
3actornetwork_action_biasadd_readvariableop_resource:	Ђ
identityИҐ*ActorNetwork/action/BiasAdd/ReadVariableOpҐ)ActorNetwork/action/MatMul/ReadVariableOpҐ3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpҐ2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOph
ActorNetwork/CastCastobservation*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Дk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€Д  Ц
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Д∞
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
ДА*
dtype0√
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А≠
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ќ
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЛ
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0—
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype0”
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЮ
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
АЂ*
dtype0љ
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЂЫ
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:Ђ*
dtype0≥
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ђy
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ђo
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   +  Щ
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?К
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Г
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Ђw
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:€€€€€€€€€Ђ[
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
value	B : Г
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
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
valueB б
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
value	B : Ь
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:–
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ]
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
valueB Ж
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
valueB:¬
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
valueB і
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
value	B : С
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€Ђy
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
valueB: 
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
value	B : з
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ј
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ`
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB 2      р?Ђ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ЂX
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB 2      рњН
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђn
IdentityIdentityclip_by_value:z:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђи
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€Д: : : : : : : : 2X
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
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:UQ
(
_output_shapes
:€€€€€€€€€Д
%
_user_specified_nameobservation
ё
&
$__inference_signature_wrapper_360126ц
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
GPU2*0J 8В *2
f-R+
)__inference_function_with_signature_23792*(
_construction_contextkEagerRuntime*
_input_shapes 
Љ
5
#__inference_get_initial_state_24042

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
“
+
)__inference_function_with_signature_23792е
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
GPU2*0J 8В *!
fR
__inference_<lambda>_690*(
_construction_contextkEagerRuntime*
_input_shapes 
шr
ђ	
'__inference_polymorphic_action_fn_23975
time_step_step_type
time_step_reward
time_step_discount
time_step_observationO
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
ДАK
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	АF
2actornetwork_action_matmul_readvariableop_resource:
АЂB
3actornetwork_action_biasadd_readvariableop_resource:	Ђ
identityИҐ*ActorNetwork/action/BiasAdd/ReadVariableOpҐ)ActorNetwork/action/MatMul/ReadVariableOpҐ3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpҐ2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpr
ActorNetwork/CastCasttime_step_observation*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Дk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€Д  Ц
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Д∞
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
ДА*
dtype0√
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А≠
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ќ
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЛ
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0—
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype0”
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЮ
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
АЂ*
dtype0љ
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЂЫ
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:Ђ*
dtype0≥
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ђy
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ђo
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   +  Щ
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?К
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Г
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Ђw
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:€€€€€€€€€Ђ[
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
value	B : Г
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
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
valueB б
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
value	B : Ь
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:–
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ]
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
valueB Ж
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
valueB:¬
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
valueB і
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
value	B : С
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€Ђy
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
valueB: 
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
value	B : з
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ј
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ`
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB 2      р?Ђ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ЂX
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB 2      рњН
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђn
IdentityIdentityclip_by_value:z:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђи
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€Д: : : : : : : : 2X
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
:€€€€€€€€€
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nametime_step/discount:_[
(
_output_shapes
:€€€€€€€€€Д
/
_user_specified_nametime_step/observation
≈
6
$__inference_signature_wrapper_360114

batch_sizeЕ
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
GPU2*0J 8В *2
f-R+
)__inference_function_with_signature_23769*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
£)
г
"__inference__traced_restore_360218
file_prefix#
assignvariableop_variable:	 J
6assignvariableop_1_actornetwork_input_mlp_dense_kernel:
ДАC
4assignvariableop_2_actornetwork_input_mlp_dense_bias:	АL
8assignvariableop_3_actornetwork_input_mlp_dense_kernel_1:
ААE
6assignvariableop_4_actornetwork_input_mlp_dense_bias_1:	АL
8assignvariableop_5_actornetwork_input_mlp_dense_kernel_2:
ААE
6assignvariableop_6_actornetwork_input_mlp_dense_bias_2:	АA
-assignvariableop_7_actornetwork_action_kernel:
АЂ:
+assignvariableop_8_actornetwork_action_bias:	Ђ
identity_10ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0* 
valueјBљ
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B –
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:Д
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_1AssignVariableOp6assignvariableop_1_actornetwork_input_mlp_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_2AssignVariableOp4assignvariableop_2_actornetwork_input_mlp_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_3AssignVariableOp8assignvariableop_3_actornetwork_input_mlp_dense_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_4AssignVariableOp6assignvariableop_4_actornetwork_input_mlp_dense_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_5AssignVariableOp8assignvariableop_5_actornetwork_input_mlp_dense_kernel_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_6AssignVariableOp6assignvariableop_6_actornetwork_input_mlp_dense_bias_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_7AssignVariableOp-assignvariableop_7_actornetwork_action_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_8AssignVariableOp+assignvariableop_8_actornetwork_action_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Х

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: V
Identity_10IdentityIdentity_9:output:0^NoOp_1*
T0*
_output_shapes
: Г
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
Љ
5
#__inference_get_initial_state_23768

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
ƒ
н
$__inference_signature_wrapper_360109
discount
observation

reward
	step_type
unknown:
ДА
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АЂ
	unknown_6:	Ђ
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_function_with_signature_23736}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:€€€€€€€€€:€€€€€€€€€Д:€€€€€€€€€:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:WS
(
_output_shapes
:€€€€€€€€€Д
'
_user_specified_name0/observation:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:PL
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type
т
_
__inference_<lambda>_687!
readvariableop_resource:	 
identity	ИҐReadVariableOp^
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
®r
М	
'__inference_polymorphic_action_fn_23717
	time_step
time_step_1
time_step_2
time_step_3O
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:
ДАK
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:	АQ
=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource:
ААM
>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource:	АF
2actornetwork_action_matmul_readvariableop_resource:
АЂB
3actornetwork_action_biasadd_readvariableop_resource:	Ђ
identityИҐ*ActorNetwork/action/BiasAdd/ReadVariableOpҐ)ActorNetwork/action/MatMul/ReadVariableOpҐ3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpҐ5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpҐ2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpҐ4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOph
ActorNetwork/CastCasttime_step_3*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Дk
ActorNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€Д  Ц
ActorNetwork/flatten/ReshapeReshapeActorNetwork/Cast:y:0#ActorNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Д∞
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource* 
_output_shapes
:
ДА*
dtype0√
#ActorNetwork/input_mlp/dense/MatMulMatMul%ActorNetwork/flatten/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А≠
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ќ
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЛ
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0—
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аі
4ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype0”
%ActorNetwork/input_mlp/dense/MatMul_2MatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:0<ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А±
5ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_2_readvariableop_resource*
_output_shapes	
:А*
dtype0‘
&ActorNetwork/input_mlp/dense/BiasAdd_2BiasAdd/ActorNetwork/input_mlp/dense/MatMul_2:product:0=ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
#ActorNetwork/input_mlp/dense/Relu_2Relu/ActorNetwork/input_mlp/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЮ
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource* 
_output_shapes
:
АЂ*
dtype0љ
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_2:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЂЫ
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes	
:Ђ*
dtype0≥
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ђy
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ђo
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   +  Щ
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?К
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ЂW
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Г
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Ђw
ActorNetwork/Cast_1CastActorNetwork/add:z:0*

DstT0*

SrcT0*,
_output_shapes
:€€€€€€€€€Ђ[
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
value	B : Г
9Deterministic/mode/Deterministic/mean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;Deterministic/mode/Deterministic/mean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
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
valueB б
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
value	B : Ь
,Deterministic/mode/Deterministic/mean/concatConcatV28Deterministic/mode/Deterministic/mean/BroadcastArgs:r0:0>Deterministic/mode/Deterministic/mean/concat/values_1:output:0:Deterministic/mode/Deterministic/mean/concat/axis:output:0*
N*
T0*
_output_shapes
:–
1Deterministic/mode/Deterministic/mean/BroadcastToBroadcastToActorNetwork/Cast_1:y:05Deterministic/mode/Deterministic/mean/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ]
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
valueB Ж
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
valueB:¬
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
valueB і
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
value	B : С
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
"Deterministic_1/sample/BroadcastToBroadcastTo:Deterministic/mode/Deterministic/mean/BroadcastTo:output:0&Deterministic_1/sample/concat:output:0*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€Ђy
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
valueB: 
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
value	B : з
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ј
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ`
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB 2      р?Ђ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ЂX
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB 2      рњН
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђn
IdentityIdentityclip_by_value:z:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђи
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_2/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€Д: : : : : : : : 2X
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
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:SO
(
_output_shapes
:€€€€€€€€€Д
#
_user_specified_name	time_step
ў
d
$__inference_signature_wrapper_360122
unknown:	 
identity	ИҐStatefulPartitionedCallЪ
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
GPU2*0J 8В *2
f-R+
)__inference_function_with_signature_23781^
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
Y

__inference_<lambda>_690*(
_construction_contextkEagerRuntime*
_input_shapes 
Ќ
i
)__inference_function_with_signature_23781
unknown:	 
identity	ИҐStatefulPartitionedCallЙ
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
GPU2*0J 8В *!
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
«
т
)__inference_function_with_signature_23736
	step_type

reward
discount
observation
unknown:
ДА
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:
АА
	unknown_4:	А
	unknown_5:
АЂ
	unknown_6:	Ђ
identityИҐStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *0
f+R)
'__inference_polymorphic_action_fn_23717}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€Д: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:WS
(
_output_shapes
:€€€€€€€€€Д
'
_user_specified_name0/observation
ƒ
;
)__inference_function_with_signature_23769

batch_size€
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
GPU2*0J 8В *,
f'R%
#__inference_get_initial_state_23768*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"ВL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*“
action«
4

0/discount&
action_0_discount:0€€€€€€€€€
?
0/observation.
action_0_observation:0€€€€€€€€€Д
0
0/reward$
action_0_reward:0€€€€€€€€€
6
0/step_type'
action_0_step_type:0€€€€€€€€€H
action>
StatefulPartitionedCall:0€€€€€€€€€€€€€€€€€€Ђtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict: _
Ќ
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
ДА 2#ActorNetwork/input_mlp/dense/kernel
2:0А 2!ActorNetwork/input_mlp/dense/bias
9:7
АА 2#ActorNetwork/input_mlp/dense/kernel
2:0А 2!ActorNetwork/input_mlp/dense/bias
9:7
АА 2#ActorNetwork/input_mlp/dense/kernel
2:0А 2!ActorNetwork/input_mlp/dense/bias
0:.
АЂ 2ActorNetwork/action/kernel
):'Ђ 2ActorNetwork/action/bias
1
ref
1"
trackable_tuple_wrapper
2
_actor_network"
_generic_user_object
ґ
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
≠
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
•
 	variables
!trainable_variables
"regularization_losses
#	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

kernel
bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

kernel
	bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

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
ї

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
≠
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
≠
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
≠
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
≠
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
≠
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
Й2Ж
'__inference_polymorphic_action_fn_23886
'__inference_polymorphic_action_fn_23975±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ж2г
-__inference_polymorphic_distribution_fn_24039±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
#__inference_get_initial_state_24042¶
Э≤Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЃBЂ
__inference_<lambda>_690"О
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЃBЂ
__inference_<lambda>_687"О
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
тBп
$__inference_signature_wrapper_360109
0/discount0/observation0/reward0/step_type"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќBЋ
$__inference_signature_wrapper_360114
batch_size"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
јBљ
$__inference_signature_wrapper_360122"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
јBљ
$__inference_signature_wrapper_360126"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2ев
ў≤’
FullArgSpecM
argsEЪB
jself
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ
Ґ 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
и2ев
ў≤’
FullArgSpecM
argsEЪB
jself
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ
Ґ 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 7
__inference_<lambda>_687Ґ

Ґ 
™ "К 	0
__inference_<lambda>_690Ґ

Ґ 
™ "™ P
#__inference_get_initial_state_24042)"Ґ
Ґ
К

batch_size 
™ "Ґ ю
'__inference_polymorphic_action_fn_23886“	
яҐџ
”Ґѕ
«≤√
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€5
observation&К#
observation€€€€€€€€€Д
Ґ 
™ "d≤a

PolicyStep8
action.К+
action€€€€€€€€€€€€€€€€€€Ђ
stateҐ 
infoҐ ¶
'__inference_polymorphic_action_fn_23975ъ	
ЗҐГ
ыҐч
п≤л
TimeStep6
	step_type)К&
time_step/step_type€€€€€€€€€0
reward&К#
time_step/reward€€€€€€€€€4
discount(К%
time_step/discount€€€€€€€€€?
observation0К-
time_step/observation€€€€€€€€€Д
Ґ 
™ "d≤a

PolicyStep8
action.К+
action€€€€€€€€€€€€€€€€€€Ђ
stateҐ 
infoҐ э
-__inference_polymorphic_distribution_fn_24039Ћ	
яҐџ
”Ґѕ
«≤√
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€5
observation&К#
observation€€€€€€€€€Д
Ґ 
™ "№≤Ў

PolicyStepЃ
action£ТЯ—ҐЌ
`
T™Q

atolК 
-
loc&К#€€€€€€€€€€€€€€€€€€Ђ

rtolК 
L™I

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
Ґ
j
parameters
Ґ 
Ґ
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec 
stateҐ 
infoҐ ќ
$__inference_signature_wrapper_360109•	
ўҐ’
Ґ 
Ќ™…
.

0/discount К

0/discount€€€€€€€€€
9
0/observation(К%
0/observation€€€€€€€€€Д
*
0/rewardК
0/reward€€€€€€€€€
0
0/step_type!К
0/step_type€€€€€€€€€"=™:
8
action.К+
action€€€€€€€€€€€€€€€€€€Ђ_
$__inference_signature_wrapper_36011470Ґ-
Ґ 
&™#
!

batch_sizeК

batch_size "™ X
$__inference_signature_wrapper_3601220Ґ

Ґ 
™ "™

int64К
int64 	<
$__inference_signature_wrapper_360126Ґ

Ґ 
™ "™ 