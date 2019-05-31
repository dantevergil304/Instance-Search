def extendBB(org_img_size, left, top, right, bottom, ratio=0.3):
    # Params:
    # - org_img_size: a tuple of (height, width)
    width = right - left
    height = bottom - top

    new_width = width * (1 + ratio)
    new_height = height * (1 + ratio)

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    return max(0, int(center_x - new_width/2)), max(0, int(center_y - new_height/2)), min(org_img_size[1], int(center_x + new_width/2)), min(org_img_size[1], int(center_y + new_height/2))
