import onnx
from onnx import checker

#320x160 模型结构深度改了，对应的节点与160x128不一样了，需要罗文峰根据模型结构，给出哪些节点需要重新修改
def reorder_onnx_nodes(input_path, output_path, new_order):
    # 加载 ONNX 模型
    model = onnx.load(input_path)
    graph = model.graph

    # 提取所有节点并建立名称到节点的映射
    node_dict = {node.name: node for node in graph.node}
    
    # 定义完整的节点顺序列表，包含所有节点
    # 清空当前图中的节点
    del graph.node[:]

    # 按照新顺序添加节点
    for node_name in new_order:
        if node_name in node_dict:
            graph.node.append(node_dict[node_name])
        else:
            print(f"警告：节点 {node_name} 未在模型中找到，已跳过")

    # 保存修改后的模型
    onnx.save(model, output_path)

    # 验证模型
    try:
        checker.check_model(model)
        print(f"模型已成功保存到 {output_path}，且通过验证")
    except checker.ValidationError as e:
        print(f"模型验证失败：{e}")

    # 打印变换后的模型节点信息
    #print(f"模型转换后的节点信息：")
    #graph = model.graph
    #nodes = {node.name: node for node in graph.node}
    #for node_name in nodes:
    #    print(node_name)


# 使用示例
if __name__ == "__main__":
    # input_onnx_path = "pt/best_yx_c2f_scdowm_model_192x192_nopsram_V3_with_intermediate.onnx"  # 输入模型路径
    # output_onnx_path = "pt/best_yx_c2f_scdowm_model_192x192_nopsram_V3_with_intermediate_reordered_model.onnx"  # 输出模型路径
    input_onnx_path = "pt/best_yx_c2f_scdowm_model_192x192_nopsram_best_06_with_intermediate.onnx"  # 输入模型路径
    output_onnx_path = "pt/best_yx_c2f_scdowm_model_192x192_nopsram_best_06_with_intermediate_reordered_model.onnx"  # 输出模型路径


    # 1. 加载 ONNX 模型
    model = onnx.load(input_onnx_path)
    graph = model.graph
    nodes = {node.name: node for node in graph.node}
    for node_name in nodes:
        print(node_name)

    value_info = {vi.name: vi for vi in graph.value_info}

    reorder_onnx_nodes(input_onnx_path, output_onnx_path)
